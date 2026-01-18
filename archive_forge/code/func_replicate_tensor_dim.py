from typing import cast, List, Optional, Sequence, Tuple
import torch
from torch.distributed._tensor._utils import compute_local_shape
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.common_rules import pointwise_rule
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def replicate_tensor_dim(placements: Sequence[Placement], dim: int) -> Tuple[Placement, ...]:
    """Force the given tensor dimension to be replicated."""
    return tuple((Replicate() if p.is_partial() or (isinstance(p, Shard) and p.dim == dim) else p for p in placements))