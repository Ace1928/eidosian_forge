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
@register_meta([aten.reflection_pad3d_backward.default, aten.reflection_pad3d_backward.grad_input, aten.replication_pad3d_backward.default, aten.replication_pad3d_backward.grad_input])
@out_wrapper('grad_input')
def meta_pad3d_backward(grad_output, input, padding):
    torch._check(len(padding) == 6, lambda: 'padding size is expected to be 6')
    assert input.ndim > 3
    assert grad_output.ndim == input.ndim
    dim_w = 3
    dim_h = 2
    dim_d = 1
    if input.ndim == 5:
        dim_w += 1
        dim_h += 1
        dim_d += 1
    pad_l, pad_r, pad_t, pad_b, pad_f, pad_bk = padding
    input_d = input.size(dim_d)
    input_h = input.size(dim_h)
    input_w = input.size(dim_w)
    output_d = input_d + pad_f + pad_bk
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r
    torch._check(output_w == grad_output.size(dim_w), lambda: f'grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}')
    torch._check(output_h == grad_output.size(dim_h), lambda: f'grad_output height unexpected. Expected: {output_h}, Got: {grad_output.size(dim_h)}')
    torch._check(output_d == grad_output.size(dim_d), lambda: f'grad_output depth unexpected. Expected: {output_d}, Got: {grad_output.size(dim_d)}')
    return input.new_empty(input.shape)