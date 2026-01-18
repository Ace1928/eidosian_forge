from typing import List, Tuple
import torch
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def linear_pointwise_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    Linear pointwise operators can propagate pending reductions.
    For example, c = add(a, b); if a is pending sum, then c will be
    pending sum as well without any communication overhead.
    """
    return pointwise_strategy(mesh, op_schema, linearity=True)