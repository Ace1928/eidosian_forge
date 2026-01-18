from dataclasses import dataclass
from typing import Callable, cast, Dict, Iterable, Optional, Sequence, Set, Tuple, Union
import torch
from torch import Tensor
from torch._subclasses.fake_tensor import unset_fake_temporarily
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.api import Shard
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import DTensorSpec, Placement, Replicate
from torch.fx.experimental.proxy_tensor import disable_proxy_modes_tracing
def dim_transpose(ndim: int, dim1: int, dim2: int) -> DimMap:
    dim1 = normalize_dim(dim1, ndim)
    dim2 = normalize_dim(dim2, ndim)
    assert dim1 < ndim
    assert dim2 < ndim
    dimmap = [InputDim(i) for i in range(ndim)]
    swapdim = dimmap[dim1]
    dimmap[dim1] = dimmap[dim2]
    dimmap[dim2] = swapdim
    return tuple(dimmap)