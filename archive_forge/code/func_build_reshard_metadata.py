import copy
from typing import List, Tuple
import torch
import torch.distributed as dist
from torch._C._distributed_c10d import (
import torch.distributed._shard.sharding_spec as shard_spec
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed.nn.functional import (
from torch.distributed._shard.metadata import ShardMetadata
from .shard import Shard
def build_reshard_metadata(st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, world_size: int) -> Tuple[List[ShardMetadata], List[int]]:
    """
    Based the given sharding spec, we calculate the offset and local shard size.
    We then build a ShardMetadata on top of the calculation result.

    Args:
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded.
        world_size (int): number of ranks.

    Returns:
        A Tuple of the followings:
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
            A List[int] which contains the ranks in the order of placement.
    """
    shard_dim = int(sharding_spec.dim)
    shards_metadata = [None] * world_size
    ranks = []
    offsets = [0] * len(st_size)
    split_size = get_split_size(st_size[shard_dim], world_size)
    for idx, placement in enumerate(sharding_spec.placements):
        ranks.append(placement.rank())
        sharded_dim_size = get_chunked_dim_size(st_size[shard_dim], split_size, idx)
        local_tensor_size = list(st_size)
        local_tensor_size[shard_dim] = sharded_dim_size
        shards_metadata[placement.rank()] = ShardMetadata(shard_offsets=copy.deepcopy(offsets), shard_sizes=local_tensor_size, placement=placement)
        offsets[shard_dim] += sharded_dim_size
    return (shards_metadata, ranks)