import functools
import operator
from typing import cast, Iterable, List, Sequence, Tuple, Union
import torch
from torch.distributed._tensor._collective_utils import redistribute_cost
from torch.distributed._tensor.api import DTensor
from torch.distributed._tensor.op_schema import OpStrategy
from torch.distributed._tensor.placement_types import (
def is_tensor_partial(spec: DTensorSpec) -> bool:
    """Return True if tensor is partial on the mesh."""
    return any((p.is_partial() for p in spec.placements))