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
def reshuffle_local_shard(local_shard: torch.Tensor, st_size: torch.Size, sharding_spec: shard_spec.ShardingSpec, resharding_spec: shard_spec.ShardingSpec, pg: ProcessGroup) -> Tuple[List[Shard], List[ShardMetadata]]:
    """
    Reshuffle the local shard directly when the reshard dim is same as the original
    sharding dim. Logically we do this in two step:
    1. To collect all shards based on original sharding spec.
    2. Reshard the tensor based on the given resharding spec.

    In reality, we consolidate the two steps into one by sending the local tensor to
    the new shard directly based on the resharding spec.

    Args:
        local_shard (Tensor): Local tensor stored in the current rank.
        st_size (torch.Size): The size of the sharded tensor.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor is sharded originally.
        resharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The
            specification describing how the tensor will be resharded.
        pg (ProcessGroup): The process group to aggregate on.

    Returns:
        A Tuple of the followings:
            A List[`Shard`] which contains the local tensor and its metadata.
            A List[`ShardMetadata`] which contains the metadata for the shard, including
                offsets, lengths and device placement.
    """
    current_rank = dist.get_rank(pg)
    world_size = dist.get_world_size(pg)
    shards_metadata, ranks = build_reshard_metadata(st_size, resharding_spec, world_size)
    reshard_dim = int(resharding_spec.dim)
    split_size = get_split_size(st_size[reshard_dim], world_size)
    input_split_sizes = [0] * world_size
    idx = get_idx_from_placements(sharding_spec.placements, current_rank)
    new_rank = resharding_spec.placements[idx].rank()
    input_split_sizes[new_rank] = local_shard.size(reshard_dim)
    output_split_sizes = [0] * world_size
    new_idx = ranks.index(current_rank)
    sharded_dim_size = get_chunked_dim_size(st_size[reshard_dim], split_size, new_idx)
    output_split_sizes[new_rank] = sharded_dim_size
    local_shard = local_shard.transpose(0, reshard_dim).contiguous()
    gathered_input_size = list(local_shard.size())
    gathered_input_size[0] = sharded_dim_size
    gathered_input = torch.empty(gathered_input_size, device=local_shard.device, dtype=local_shard.dtype)
    local_shard = all_to_all_single(gathered_input, local_shard, input_split_sizes=input_split_sizes, output_split_sizes=output_split_sizes, group=pg)
    local_tensor = local_shard.transpose(0, reshard_dim).contiguous()
    local_shards = [Shard(local_tensor, shards_metadata[current_rank])]
    return (local_shards, shards_metadata)