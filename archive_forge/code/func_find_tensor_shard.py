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
def find_tensor_shard(tensor: torch.Tensor, index: MetadataIndex) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    if isinstance(tensor, ShardedTensor):
        return _find_shard(tensor, index).tensor
    if index.offset is not None:
        if index.offset == torch.Size([0] * len(tensor.size())):
            return tensor
        raise ValueError(f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'")
    return tensor