from typing import cast, List, Optional, Sequence, Tuple
import torch
import torch.distributed.distributed_c10d as c10d
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def replicate_reduction_dims(placements: Tuple[Placement, ...], reduction_dims: List[int]) -> Tuple[Placement, ...]:
    new_placements: List[Placement] = []
    for p in placements:
        if p.is_partial():
            new_placements.append(Replicate())
        elif isinstance(p, Shard) and p.dim in reduction_dims:
            new_placements.append(Replicate())
        else:
            new_placements.append(p)
    return tuple(new_placements)