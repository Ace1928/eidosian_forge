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
@register_meta(torch.ops.aten.t_)
def t_(self):
    ndims = self.ndim
    if self.is_sparse:
        sparse_dim = self.sparse_dim()
        dense_dim = self.dense_dim()
        assert sparse_dim <= 2 and dense_dim == 0, f't_ expects a tensor with <= 2 sparse and 0 dense dimensions, but got {sparse_dim} sparse and {dense_dim} dense dimensions'
    else:
        assert self.dim() <= 2, f't_ expects a tensor with <= 2 dimensions, but self is {ndims}D'
    return transpose_(self, 0, 0 if ndims < 2 else 1)