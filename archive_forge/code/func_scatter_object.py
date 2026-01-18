import os
import io
import itertools
from typing import (
import torch.distributed as dist
from .api import (
import torch
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.sharded_tensor.shard import Shard
from torch.distributed._tensor import DTensor
from .metadata import (
def scatter_object(self, object_list: Optional[List[T]]) -> T:
    """Implement functionality similar to c10d::scatter_object but without distributed enabled."""
    if self.use_dist:
        gather_result = cast(List[T], [None])
        dist.scatter_object_list(scatter_object_output_list=gather_result, scatter_object_input_list=object_list if self.is_coordinator else None, src=self.coordinator_rank, group=self.group)
        local_reply = gather_result[0]
    else:
        assert object_list is not None
        local_reply = object_list[0]
    return local_reply