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
@register_meta(aten.select.int)
def meta_select(self, dim, index):
    ndim = self.dim()
    torch._check_index(ndim != 0, lambda: 'select() cannot be applied to a 0-dim tensor.')
    dim = dim if dim >= 0 else dim + ndim
    size = self.size(dim)
    torch._check_index(not (-index > size or index >= size), lambda: f'select(): index {index} out of range for tensor of size {self.size()} at dimension {dim}')
    index = index if index >= 0 else index + size
    new_size = list(self.size())
    new_stride = list(self.stride())
    new_storage_offset = self.storage_offset() + index * new_stride[dim]
    del new_size[dim]
    del new_stride[dim]
    return self.as_strided(new_size, new_stride, new_storage_offset)