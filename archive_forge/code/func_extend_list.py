import torch
from typing import (
from torch.distributed.checkpoint.metadata import (
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DTensor
def extend_list(lst: List[STATE_DICT_ITEM], idx: int) -> None:
    while len(lst) <= idx:
        lst.append(None)