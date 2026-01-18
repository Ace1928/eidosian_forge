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
@register_meta(torch.ops.mkldnn._convolution_pointwise.default)
def meta_mkldnn_convolution_default(input_tensor, weight, bias, padding, stride, dilation, groups, attr, scalars, algorithm):
    shape_out = calc_conv_nd_return_shape(input_tensor, weight, stride, padding, dilation, False, groups, [])
    out = input_tensor.new_empty(shape_out)
    out_memory_format = torch.channels_last
    out = out.to(memory_format=out_memory_format)
    return out