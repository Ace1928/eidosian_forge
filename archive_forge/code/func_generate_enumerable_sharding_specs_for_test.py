import builtins
import torch
from torch.distributed._shard.sharding_spec import (
from torch.distributed._shard.sharding_spec._internals import (
def generate_enumerable_sharding_specs_for_test():
    return [EnumerableShardingSpec([ShardMetadata(shard_offsets=[0, 0], shard_sizes=[5, 5], placement='rank:0/cuda:0'), ShardMetadata(shard_offsets=[5, 0], shard_sizes=[5, 5], placement='rank:1/cuda:1'), ShardMetadata(shard_offsets=[0, 5], shard_sizes=[5, 5], placement='rank:2/cuda:2'), ShardMetadata(shard_offsets=[5, 5], shard_sizes=[5, 5], placement='rank:3/cuda:3')])]