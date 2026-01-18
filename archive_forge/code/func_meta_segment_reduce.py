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
@register_meta(aten.segment_reduce.default)
def meta_segment_reduce(data: Tensor, reduce: str, *, lengths: Optional[Tensor]=None, indices: Optional[Tensor]=None, offsets: Optional[Tensor]=None, axis: int=0, unsafe: bool=False, initial=None) -> Tensor:
    if indices is not None:
        raise NotImplementedError('segment_reduce(): indices based reduction is not supported yet.')

    def segment_reduce_lengths_tensor(lengths_shape):
        return torch.empty(lengths_shape + data.shape[axis + 1:], dtype=data.dtype, device='meta', memory_format=torch.contiguous_format)
    if lengths is not None:
        return segment_reduce_lengths_tensor(lengths.shape)
    if offsets is not None:
        lengths_shape = offsets.shape[:-1] + (offsets.shape[-1] - 1,)
        return segment_reduce_lengths_tensor(lengths_shape)
    raise RuntimeError('segment_reduce(): Either lengths or offsets must be defined.')