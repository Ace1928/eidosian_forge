import sys
from functools import wraps, partial
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.testing._internal.common_distributed import (
def assert_sharded_tensor_equal(self, st1, st2):
    st1_local_shards = st1.local_shards()
    st2_local_shards = st2.local_shards()
    self.assertEqual(len(st1_local_shards), len(st2_local_shards))
    for i, st1_local_shard in enumerate(st1_local_shards):
        self.assertEqual(st1_local_shard.tensor, st2_local_shards[i].tensor)
        self.assertEqual(st1_local_shard.metadata, st2_local_shards[i].metadata)
    self.assertEqual(st1.metadata(), st2.metadata())
    self.assertEqual(st1.sharding_spec(), st2.sharding_spec())
    self.assertEqual(len(st1.remote_shards()), len(st2.remote_shards()))