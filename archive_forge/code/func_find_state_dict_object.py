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
def find_state_dict_object(state_dict: STATE_DICT_TYPE, index: MetadataIndex) -> Any:
    if index.fqn not in state_dict:
        raise ValueError(f"Could not find FQN: '{index.fqn}'")
    obj = state_dict[index.fqn]
    if isinstance(obj, torch.Tensor):
        return find_tensor_shard(obj, index)
    elif index.offset is not None:
        raise ValueError(f"FQN: '{index.fqn}' is not a ShardedTensor, can't find by offset: '{index.offset}'")
    return obj