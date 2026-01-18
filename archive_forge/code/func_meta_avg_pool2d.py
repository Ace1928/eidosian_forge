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
@register_meta(aten.avg_pool2d.default)
def meta_avg_pool2d(input, kernel_size, stride=(), padding=(0,), ceil_mode=False, count_include_pad=True, divisor_override=None):

    def unpack(name, val):
        torch._check(len(val) in [1, 2], lambda: f'avg_pool2d: {name} must either be a single int, or a tuple of two ints')
        H = val[0]
        W = H if len(val) == 1 else val[1]
        return (H, W)
    kH, kW = unpack('kernel_size', kernel_size)
    torch._check(len(stride) in [0, 1, 2], lambda: 'avg_pool2d: stride must either be omitted, a single int, or a tuple of two ints')
    if len(stride) == 0:
        dH, dW = (kH, kW)
    elif len(stride) == 1:
        dH, dW = (stride[0], stride[0])
    else:
        dH, dW = unpack('stride', stride)
    padH, padW = unpack('padding', padding)
    torch._check(divisor_override is None or divisor_override != 0, lambda: 'divisor must be not zero')
    nbatch = input.size(-4) if input.dim() == 4 else 1
    nInputPlane = input.size(-3)
    inputHeight = input.size(-2)
    inputWidth = input.size(-1)
    outputHeight = pooling_output_shape(inputHeight, kH, padH, dH, 1, ceil_mode)
    outputWidth = pooling_output_shape(inputWidth, kW, padW, dW, 1, ceil_mode)
    memory_format = utils.suggest_memory_format(input)
    pool2d_shape_check(input, kH, kW, dH, dW, padH, padW, 1, 1, nInputPlane, inputHeight, inputWidth, outputHeight, outputWidth, memory_format)
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return torch.empty(size, dtype=input.dtype, device=input.device, memory_format=memory_format)