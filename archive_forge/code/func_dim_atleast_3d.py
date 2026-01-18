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
def dim_atleast_3d(ndim: int) -> DimMap:
    if ndim == 0:
        return (Singleton(), Singleton(), Singleton())
    elif ndim == 1:
        return (Singleton(), InputDim(0), Singleton())
    elif ndim == 2:
        return (InputDim(0), InputDim(1), Singleton())
    else:
        return tuple((InputDim(i) for i in range(ndim)))