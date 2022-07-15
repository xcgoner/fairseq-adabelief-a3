# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import types

import torch
import torch.optim
import torch.distributed as dist
from torch.optim.optimizer import Optimizer

from . import FairseqOptimizer, register_optimizer


import math
import torch
from torch.optim.optimizer import Optimizer

version_higher = (torch.__version__ >= "1.5.0")

from . import LegacyFairseqOptimizer, register_optimizer


@register_optimizer("a3")
class FairseqA3(LegacyFairseqOptimizer):
    def __init__(self, args, params):
        super().__init__(args)
        self._optimizer = A3(params, **self.optimizer_config)

    @staticmethod
    def add_args(parser):
        """Add optimizer-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--a3-betas', default='(0.9, 0.999, 0.999)', metavar='B',
                            help='betas for A3 optimizer')
        parser.add_argument('--a3-rho', type=float, default=0.08, metavar='R',
                            help='rho for A3 optimizer')
        parser.add_argument('--a3-eps', type=float, default=1e-16, metavar='D',
                            help='epsilon for A3 optimizer')
        parser.add_argument('--weight-decay', '--wd', default=0.0, type=float, metavar='WD',
                            help='weight decay')
        parser.add_argument('--weight-decouple', default=True, type=bool)
        parser.add_argument('--rectify', default=False, type=bool)
        # fmt: on

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        betas = eval(self.args.a3_betas)
        if len(betas) == 1:
            betas = (betas[0], betas[0], betas[0])
        elif len(betas) == 2:
            betas = (betas[0], betas[1], betas[1])
        return {
            "lr": self.args.lr[0],
            "betas": betas,
            "rho": self.args.a3_rho,
            "eps": self.args.a3_eps,
            "weight_decay": self.args.weight_decay,
        }


class A3(Optimizer):
    r"""Implements A3 algorithm. Modified from Adam in PyTorch

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        weight_decouple (boolean, optional): ( default: False) If set as True, then
            the optimizer uses decoupled weight decay as in AdamW
        fixed_decay (boolean, optional): (default: False) This is used when weight_decouple
            is set as True.
            When fixed_decay == True, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay$.
            When fixed_decay == False, the weight decay is performed as
            $W_{new} = W_{old} - W_{old} \times decay \times lr$. Note that in this case, the
            weight decay ratio decreases with learning rate (lr).
        rectify (boolean, optional): (default: False) If set as True, then perform the rectified
            update similar to RAdam

    reference: A3 Optimizer
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.999), rho=0.16, eps=1e-16,
                 weight_decay=0, amsgrad=False, weight_decouple=True, fixed_decay=False, rectify=False,
                 degenerated_to_sgd=True):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[2]))

        self.degenerated_to_sgd = degenerated_to_sgd
        if isinstance(params, (list, tuple)) and len(params) > 0 and isinstance(params[0], dict):
            for param in params:
                if 'betas' in param and (param['betas'][0] != betas[0] or param['betas'][1] != betas[1]):
                    param['buffer'] = [[None, None, None] for _ in range(10)]

        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, buffer=[[None, None, None] for _ in range(10)])
        super(A3, self).__init__(params, defaults)

        self.degenerated_to_sgd = degenerated_to_sgd
        self.weight_decouple = weight_decouple
        self.rectify = rectify
        self.fixed_decay = fixed_decay
        if self.weight_decouple:
            print('Weight decoupling enabled in A3')
            if self.fixed_decay:
                print('Weight decay fixed')
        if self.rectify:
            print('Rectification enabled in A3')
        if amsgrad:
            print('AMS enabled in A3')

    def __setstate__(self, state):
        super(A3, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                amsgrad = group['amsgrad']

                # State initialization
                state['step'] = 0
                # Exponential moving average of gradient values
                state['exp_avg'] = torch.zeros_like(p.data,
                                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                    p.data)

                # Exponential moving average of squared gradient values
                state['exp_avg_var'] = torch.zeros_like(p.data,
                                                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                    p.data)
                
                # Exponential moving average of 3rd moment
                state['exp_avg_m3'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)
                # gradient 1 step back
                state['prev_grad_1'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)
                # gradient 2 steps back
                state['prev_grad_2'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                    if version_higher else torch.zeros_like(p.data)
                
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state['max_exp_avg_var'] = torch.zeros_like(p.data,
                                                                memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p.data)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        'A3 does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                beta1, beta2, beta3 = group['betas']

                # State initialization
                if len(state) == 0:
                    state['rho_inf'] = 2.0 / (1.0 - beta2) - 1.0
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data,
                                                        memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_var'] = torch.zeros_like(p.data,
                                                            memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                        p.data)
                    # Exponential moving average of 3rd moment
                    state['exp_avg_m3'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # gradient 1 step back
                    state['prev_grad_1'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    # gradient 2 steps back
                    state['prev_grad_2'] = torch.zeros_like(p.data,memory_format=torch.preserve_format) \
                        if version_higher else torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_var'] = torch.zeros_like(p.data,
                                                                    memory_format=torch.preserve_format) if version_higher else torch.zeros_like(
                            p.data)

                # get current state variable
                exp_avg, exp_avg_var, exp_avg_m3 = state['exp_avg'], state['exp_avg_var'], state['exp_avg_m3']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Update first and second moment running average
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                bias_corrected_exp_avg = exp_avg / bias_correction1
                grad_residual = grad - exp_avg
                exp_avg_var.mul_(beta2).addcmul_( grad_residual, grad_residual, value=1 - beta2)

                # Update 3rd moment running average
                prev_grad_1, prev_grad_2 = state['prev_grad_1'], state['prev_grad_2']
                if state['step'] % 2 == 0:
                    prev_grad_1, prev_grad_2 = prev_grad_2, prev_grad_1
                bias_correction3 = 1.
                if state['step'] >= 3:
                    bias_correction3 = 1 - beta3 ** (state['step'] - 2)
                    m3 = grad + prev_grad_2 - prev_grad_1 * 2
                    # m3 = (bias_corrected_exp_avg + prev_grad_2 - prev_grad_1 * 2) * group['rho']
                    exp_avg_m3.mul_(beta3).add_(m3, alpha=(1 - beta3) * group['rho'])
                # prev_grad_2.copy_(grad)
                prev_grad_2.copy_(bias_corrected_exp_avg)
                bias_corrected_exp_avg_m3 = exp_avg_m3 / bias_correction3

                if amsgrad:
                    max_exp_avg_var = state['max_exp_avg_var']
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_var, exp_avg_var.add_(group['eps']), out=max_exp_avg_var)

                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_var.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_var.add_(group['eps']).sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                if not self.rectify:
                    # perform weight decay, check if decoupled weight decay
                    if self.weight_decouple:
                        if not self.fixed_decay:
                            p.data.mul_(1.0 - group['lr'] * group['weight_decay'])
                        else:
                            p.data.mul_(1.0 - group['weight_decay'])
                    else:
                        if group['weight_decay'] != 0:
                            grad.add_( p.data, alpha=group['weight_decay'])

                    # Default update
                    step_size = group['lr']
                    p.data.addcdiv_( bias_corrected_exp_avg + bias_corrected_exp_avg_m3, denom, value=-step_size)

                else:  # Rectified update
                    buffered = group['buffer'][int(state['step'] % 10)]
                    if state['step'] == buffered[0]:
                        N_sma, step_size = buffered[1], buffered[2]
                    else:
                        buffered[0] = state['step']
                        beta2_t = beta2 ** state['step']
                        N_sma_max = 2 / (1 - beta2) - 1
                        N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)
                        buffered[1] = N_sma

                        # more conservative since it's an approximated value
                        if N_sma >= 5:
                            step_size = math.sqrt(
                                (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (
                                        N_sma_max - 2)) / (1 - beta1 ** state['step'])
                        elif self.degenerated_to_sgd:
                            step_size = 1.0 / (1 - beta1 ** state['step'])
                        else:
                            step_size = -1
                        buffered[2] = step_size

                    # more conservative since it's an approximated value
                    p_data_fp32 = p.data#.float()
                    if N_sma >= 5:
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_( p_data_fp32, alpha = -group['weight_decay'] * group['lr'])
                        denom = exp_avg_var.sqrt().add_(group['eps'])
                        p_data_fp32.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
                        #p.data.copy_(p_data_fp32)
                    elif step_size > 0:
                        if group['weight_decay'] != 0:
                            p_data_fp32.add_( p_data_fp32, alpha=-group['weight_decay'] * group['lr'])
                        p_data_fp32.add_( exp_avg, alpha=-step_size * group['lr'])
                        #p.data.copy_(p_data_fp32)

        return loss

