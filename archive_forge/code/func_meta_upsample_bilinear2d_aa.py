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
@register_meta(aten._upsample_bilinear2d_aa.default)
def meta_upsample_bilinear2d_aa(input, output_size, align_corners, scales_h=None, scales_w=None):
    full_output_size = upsample_common_check(input.size(), output_size, num_spatial_dims=2)
    torch._check(input.numel() != 0 or all((size > 0 for size in input.size()[1:])), lambda: f'Non-empty 4D data tensor expected but got a tensor with sizes {input.size()}')
    return input.new_empty(full_output_size).to(memory_format=utils.suggest_memory_format(input))