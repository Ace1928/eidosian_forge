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
@register_meta([aten.baddbmm.default, aten.baddbmm.out])
@out_wrapper()
def meta_baddbmm(self, batch1, batch2, *, beta=1, alpha=1):
    dim1 = batch1.size(0)
    dim2 = batch1.size(1)
    dim3 = batch2.size(2)
    self = self.expand((dim1, dim2, dim3))
    torch._check(batch1.dim() == 3, lambda: 'batch1 must be a 3D tensor')
    torch._check(batch2.dim() == 3, lambda: 'batch2 must be a 3D tensor')
    torch._check(self.dtype == batch1.dtype == batch2.dtype, lambda: f'Input dtypes must be the same, got: input: {self.dtype}, batch1: {batch1.dtype}, batch2: {batch2.dtype}')
    batch1_sizes = batch1.shape
    batch2_sizes = batch2.shape
    bs = batch1_sizes[0]
    contraction_size = batch1_sizes[2]
    torch._check(batch2_sizes[0] == bs and batch2_sizes[1] == contraction_size, lambda: f'Expected size for first two dimensions of batch2 tensor to be: [{bs}, {contraction_size}] but got: [{batch2_sizes[0]}, {batch2_sizes[1]}].')
    return self.new_empty(self.size())