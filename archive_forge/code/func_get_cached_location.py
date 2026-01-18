from dataclasses import dataclass
from typing import Optional, Tuple
import ray
from .common import NodeIdStr
from ray.data._internal.memory_tracing import trace_deallocation
from ray.data.block import Block, BlockMetadata
from ray.data.context import DataContext
from ray.types import ObjectRef
def get_cached_location(self) -> Optional[NodeIdStr]:
    """Return a location for this bundle's data, if possible.

        Caches the resolved location so multiple calls to this are efficient.
        """
    if self._cached_location is None:
        ref = self.blocks[0][0]
        locs = ray.experimental.get_object_locations([ref])
        nodes = locs[ref]['node_ids']
        if nodes:
            self._cached_location = nodes[0]
        else:
            self._cached_location = ''
    if self._cached_location:
        return self._cached_location
    else:
        return None