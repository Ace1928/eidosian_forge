import functools
import operator
from typing import cast, Iterable, List, Sequence, Tuple, Union
import torch
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.op_schema import OpStrategy
from torch.distributed._tensor.placement_types import (
def map_placements_after_broadcast(placements: Tuple[Placement, ...], shape: torch.Size, broadcast_dims_map: List[int]) -> Tuple[Placement, ...]:
    """Map each placement based on the output shape after broadcast."""
    new_placements: List[Placement] = []
    for placement in placements:
        if isinstance(placement, (Replicate, _Partial)):
            new_placements.append(placement)
        else:
            assert isinstance(placement, Shard)
            shard_dim = normalize_dim(placement.dim, len(shape))
            new_shard_dim = broadcast_dims_map[shard_dim]
            if new_shard_dim != -1:
                new_placements.append(Shard(new_shard_dim))
            else:
                new_placements.append(Replicate())
    return tuple(new_placements)