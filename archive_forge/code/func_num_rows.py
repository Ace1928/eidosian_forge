from dataclasses import dataclass
from typing import Optional, Tuple
import ray
from .common import NodeIdStr
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def num_rows(self) -> Optional[int]:
    """Number of rows present in this bundle, if known."""
    total = 0
    for b in self.blocks:
        if b[1].num_rows is None:
            return None
        else:
            total += b[1].num_rows
    return total