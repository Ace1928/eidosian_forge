import torch
from torch.distributed._shard.metadata import ShardMetadata
from typing import Sequence
def narrow_tensor(tensor: torch.Tensor, metadata: ShardMetadata) -> torch.Tensor:
    """
    Narrow the tensor according to the metadata
    """
    return narrow_tensor_by_index(tensor, metadata.shard_offsets, metadata.shard_sizes)