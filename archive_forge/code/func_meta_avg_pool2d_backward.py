import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(aten.avg_pool2d_backward.default)
def meta_avg_pool2d_backward(gradOutput_, input, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override):
    torch._check(len(kernel_size) == 1 or len(kernel_size) == 2, lambda: 'avg_pool2d: kernel_size must either be a single int, or a tuple of two ints')
    kH = kernel_size[0]
    kW = kH if len(kernel_size) == 1 else kernel_size[1]
    torch._check(len(stride) == 0 or len(stride) == 1 or len(stride) == 2, lambda: 'avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints')
    dH = kH if len(stride) == 0 else stride[0]
    dW = kW if len(stride) == 0 else dH if len(stride) == 1 else stride[1]
    torch._check(len(padding) == 1 or len(padding) == 2, lambda: 'avg_pool2d: padding must either be a single int, or a tuple of two ints')
    padH = padding[0]
    padW = padH if len(padding) == 1 else padding[1]
    torch._check(divisor_override is None or divisor_override != 0, lambda: 'divisor must be not zero')
    input_size = input.shape
    nbatch = input_size[-4] if input.dim() == 4 else 1
    nInputPlane = input_size[-3]
    inputHeight = input_size[-2]
    inputWidth = input_size[-1]
    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)
    mem_format = utils.suggest_memory_format(input)
    avg_pool2d_backward_shape_check(input, gradOutput_, nbatch, kH, kW, dH, dW, padH, padW, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, mem_format)
    return torch.empty(input_size, dtype=input.dtype, device=input.device, memory_format=mem_format)