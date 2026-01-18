import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional
from .expanded_weights_utils import \
def conv_unfold_weight_grad_sample(input, grad_output, weight_shape, kernel_size, stride, padding, dilation, groups, func):
    n = input.shape[0]
    in_channels = input.shape[1]
    unfold_func = conv_picker(func, lambda: F.unfold(input.unsqueeze(-2), kernel_size=(1, kernel_size[0]), dilation=(1, dilation[0]), padding=(0, padding[0]), stride=(1, stride[0])), lambda: F.unfold(input, kernel_size, dilation=dilation, padding=padding, stride=stride), lambda: unfold3d(input, kernel_size, padding, stride, dilation))
    input = unfold_func()
    grad_output = grad_output.reshape(n, -1, input.shape[-1])
    weight_grad_sample = torch.einsum('noq,npq->nop', grad_output, input)
    weight_grad_sample = weight_grad_sample.view(n, groups, -1, groups, int(in_channels / groups), np.prod(kernel_size))
    weight_grad_sample = torch.einsum('ngrg...->ngr...', weight_grad_sample).contiguous()
    shape = [n] + list(weight_shape)
    weight_grad_sample = weight_grad_sample.view(shape)
    return weight_grad_sample