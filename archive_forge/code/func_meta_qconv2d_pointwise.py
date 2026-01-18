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
@register_meta(torch.ops.onednn.qconv2d_pointwise.default)
def meta_qconv2d_pointwise(x, x_scale, x_zp, w, w_scale, w_zp, bias, stride, padding, dilation, groups, output_scale, output_zero_point, output_dtype, attr, scalars, algorithm):
    shape_out = calc_conv_nd_return_shape(x, w, stride, padding, dilation, False, groups, None)
    assert output_dtype in [torch.float32, torch.bfloat16]
    out = x.new_empty(shape_out, dtype=output_dtype)
    out = out.to(memory_format=torch.channels_last)
    return out