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
def gather_shape_check(self, dim, index):
    self_dims = max(self.dim(), 1)
    index_dims = max(index.dim(), 1)
    torch._check(self_dims == index_dims, lambda: 'Index tensor must have the same number of dimensions as input tensor')
    for i in range(self_dims):
        if i != dim:
            torch._check(ensure_nonempty_size(index, i) <= ensure_nonempty_size(self, i), lambda: f'Size does not match at dimension {i} expected index {index.shape}' + f' to be smaller than self {self.shape} apart from dimension {dim}')