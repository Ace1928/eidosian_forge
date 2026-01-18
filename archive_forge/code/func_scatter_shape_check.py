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
def scatter_shape_check(self, dim, index, src_opt=None):
    if index.numel() == 0:
        return
    torch._check(ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()), lambda: 'Index tensor must have the same number of dimensions as self tensor')
    is_wrong_shape = False
    self_dims = ensure_nonempty_dim(self.dim())
    for d in range(self_dims):
        index_d_size = ensure_nonempty_size(index, d)
        if d == dim:
            continue
        if index_d_size > ensure_nonempty_size(self, d):
            is_wrong_shape = True
            break
    if not is_wrong_shape and src_opt is not None:
        for d in range(self_dims):
            index_d_size = ensure_nonempty_size(index, d)
            if index_d_size > ensure_nonempty_size(src_opt, d):
                is_wrong_shape = True
                break
    if src_opt is not None:
        torch._check(ensure_nonempty_dim(self.dim()) == ensure_nonempty_dim(index.dim()), lambda: 'Index tensor must have the same number of dimensions as self tensor')
        torch._check(not is_wrong_shape, lambda: f'Expected index {index.shape} to be smaller than self {self.shape}' + f' apart from dimension {dim} and to be smaller than src {src_opt.shape}')
    else:
        torch._check(not is_wrong_shape, lambda: f'Expected index {index.shape} to be smaller than self {self.shape}' + f' apart from dimension {dim}')