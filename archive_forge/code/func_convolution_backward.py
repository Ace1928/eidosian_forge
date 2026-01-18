import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
@register_decomposition([aten.convolution_backward])
def convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask):
    if not output_mask[2] or grad_output.device.type != 'cuda':
        return NotImplemented
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    grad_inp, grad_weight, _ = aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, [output_mask[0], output_mask[1], False])
    return (grad_inp, grad_weight, grad_bias)