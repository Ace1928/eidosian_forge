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
@register_meta([aten.reflection_pad2d_backward.default, aten.reflection_pad2d_backward.grad_input, aten.replication_pad2d_backward.default, aten.replication_pad2d_backward.grad_input])
@out_wrapper('grad_input')
def meta_pad2d_backward(grad_output, self, padding):
    dim_w = 2
    dim_h = 1
    dim_plane = 0
    nbatch = 1
    self_shape = self.shape
    if self.dim() == 4:
        nbatch = self_shape[0]
        dim_w += 1
        dim_h += 1
        dim_plane += 1
    pad_l, pad_r, pad_t, pad_b = padding
    nplane = self_shape[dim_plane]
    input_h = self_shape[dim_h]
    input_w = self_shape[dim_w]
    output_h = input_h + pad_t + pad_b
    output_w = input_w + pad_l + pad_r
    torch._check(output_w == grad_output.size(dim_w), lambda: f'grad_output width unexpected. Expected: {output_w}, Got: {grad_output.size(dim_w)}')
    torch._check(output_h == grad_output.size(dim_h), lambda: f'grad_output height unexpected. Expected: {output_h}, Got: {grad_output.size(dim_h)}')
    return self.new_empty(self.shape)