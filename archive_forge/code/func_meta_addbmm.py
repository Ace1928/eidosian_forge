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
@register_meta([aten.addbmm.default, aten.addbmm.out])
@out_wrapper()
def meta_addbmm(self, batch1, batch2, *, beta=1, alpha=1):
    dim1 = batch1.size(1)
    dim2 = batch2.size(2)
    self = self.expand((dim1, dim2))
    torch._check(batch1.dim() == 3, lambda: 'batch1 must be a 3D tensor')
    torch._check(batch2.dim() == 3, lambda: 'batch2 must be a 3D tensor')
    torch._check(batch1.size(0) == batch2.size(0), lambda: f'batch1 and batch2 must have same number of batches, got {batch1.size(0)} and {batch2.size(0)}')
    torch._check(batch1.size(2) == batch2.size(1), lambda: f'Incompatible matrix sizes for bmm ({batch1.size(1)}x{batch1.size(2)} and {batch2.size(1)}x{batch2.size(2)})')
    torch._check(self.size(0) == dim1 and self.size(1) == dim2, lambda: 'self tensor does not match matmul output shape')
    return self.new_empty(self.size())