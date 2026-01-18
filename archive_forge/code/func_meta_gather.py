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
@register_meta(aten.gather.default)
def meta_gather(self, dim, index, sparse_grad=False):
    wrapped_dim = maybe_wrap_dim(dim, self.dim())
    is_index_empty = index.numel() == 0
    if not is_index_empty:
        torch._check(index.dtype == torch.long, lambda: f'gather(): Expected dtype int64 for index, but got {index.dtype}')
        gather_shape_check(self, wrapped_dim, index)
    return self.new_empty(index.shape)