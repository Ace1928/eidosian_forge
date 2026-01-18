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
@register_meta([aten.index.Tensor, aten._unsafe_index.Tensor])
def meta_index_Tensor(self, indices):
    torch._check(bool(indices), lambda: 'at least one index must be provided')
    result: List[Optional[Tensor]] = []
    for i, index in enumerate(indices):
        if index is not None:
            torch._check(index.dtype in [torch.long, torch.int, torch.int8, torch.bool], lambda: 'tensors used as indices must be long, int, byte or bool tensors')
            if index.dtype in [torch.int8, torch.bool]:
                nonzero = index.nonzero()
                k = len(result)
                torch._check_index(k + index.ndim <= self.ndim, lambda: f'too many indices for tensor of dimension {self.ndim}')
                for j in range(index.ndim):
                    torch._check_index(index.shape[j] == self.shape[k + j], lambda: f'The shape of the mask {index.shape} at index {i} does not match the shape of the indexed tensor {self.shape} at index {k + j}')
                    result.append(nonzero.select(1, j))
            else:
                result.append(index)
        else:
            result.append(index)
    indices = result
    torch._check(len(indices) <= self.ndim, lambda: f'too many indices for tensor of dimension {self.ndim} (got {len(indices)})')
    import torch._refs as refs
    indices = list(refs._maybe_broadcast(*indices))
    while len(indices) < self.ndim:
        indices.append(None)
    state = 0
    has_contiguous_subspace = False
    for index in indices:
        if state == 0:
            if index is not None:
                state = 1
        elif state == 1:
            if index is None:
                state = 2
        elif index is not None:
            break
    else:
        has_contiguous_subspace = True
    if not has_contiguous_subspace:
        dims = []
        transposed_indices = []
        for i, index in enumerate(indices):
            if index is not None:
                dims.append(i)
                transposed_indices.append(index)
        for i, index in enumerate(indices):
            if index is None:
                dims.append(i)
                transposed_indices.append(index)
        self = self.permute(dims)
        indices = transposed_indices
    before_shape: List[int] = []
    after_shape: List[int] = []
    replacement_shape: List[int] = []
    for dim, index in enumerate(indices):
        if index is None:
            if replacement_shape:
                after_shape.append(self.shape[dim])
            else:
                before_shape.append(self.shape[dim])
        else:
            replacement_shape = list(index.shape)
    return self.new_empty(before_shape + replacement_shape + after_shape)