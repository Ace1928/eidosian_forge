from typing import List, Tuple
import torch
from torch.distributed._tensor.op_schema import (
from torch.distributed._tensor.ops.utils import (
from torch.distributed._tensor.placement_types import (
from torch.distributed.device_mesh import DeviceMesh
def foreach_list_linear_strategy(mesh: DeviceMesh, op_schema: OpSchema) -> StrategyType:
    """
    for each list op stratgy that supports linearity
    """
    return foreach_list_strategy(mesh, op_schema, linearity=True)