from typing import Dict, List, Optional, Tuple
import torch
import torch.optim._functional as F
from torch import Tensor
def step_param(self, param: Tensor, grad: Optional[Tensor]):
    params_with_grad = []
    grads = []
    exp_avgs = []
    exp_avg_sqs = []
    max_exp_avg_sqs = []
    state_steps: List[Tensor] = []
    has_complex = torch.is_complex(param)
    if grad is not None:
        params_with_grad.append(param)
        grads.append(grad)
    if param not in self.state:
        self.state[param] = {}
        state = self.state[param]
        state['step'] = torch.tensor(0.0)
        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
        if self.amsgrad:
            state['max_exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)
    state = self.state[param]
    exp_avgs.append(state['exp_avg'])
    exp_avg_sqs.append(state['exp_avg_sq'])
    if self.amsgrad:
        max_exp_avg_sqs.append(state['max_exp_avg_sq'])
    state_steps.append(state['step'])
    with torch.no_grad():
        F.adamw(params_with_grad, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps, amsgrad=self.amsgrad, maximize=self.maximize, beta1=self.defaults['beta1'], beta2=self.defaults['beta2'], lr=self.defaults['lr'], weight_decay=self.defaults['weight_decay'], eps=self.defaults['eps'], foreach=self.foreach, fused=self.fused, grad_scale=None, found_inf=None, has_complex=has_complex)