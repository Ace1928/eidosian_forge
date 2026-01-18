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
@register_meta(aten.convolution.default)
def meta_conv(input_tensor: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, stride: List[int], padding: List[int], dilation: List[int], is_transposed: bool, output_padding: List[int], groups: int):

    def pick_memory_format():
        if device_hint(input_tensor) == 'cuda':
            if is_channels_last(input_tensor) or is_channels_last(weight):
                return torch.channels_last
        elif is_channels_last(input_tensor):
            return torch.channels_last
        if input_tensor.is_contiguous(memory_format=torch.contiguous_format):
            return torch.contiguous_format
        elif input_tensor.is_contiguous(memory_format=torch.preserve_format):
            return torch.preserve_format
    shape_out = calc_conv_nd_return_shape(input_tensor, weight, stride, padding, dilation, is_transposed, groups, output_padding if is_transposed else None)
    out = input_tensor.new_empty(shape_out)
    out = out.to(memory_format=pick_memory_format())
    return out