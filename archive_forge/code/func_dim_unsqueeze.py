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
def dim_unsqueeze(ndim: int, dim: int) -> DimMap:
    dims = tuple((InputDim(i) for i in range(ndim)))
    if dim < 0:
        dim += ndim + 1
    return dims[:dim] + (Singleton(),) + dims[dim:]