import copy
import torch
from torch.distributed._shard.sharded_tensor import (
from ._common import (
from torch.distributed._shard.common_op_utils import _register_default_op
def sharded_clone(args, kwargs, pg):
    self_st = args[0]
    desire_memory_format = kwargs.get('memory_format', None)
    if desire_memory_format and desire_memory_format != torch.preserve_format:
        raise RuntimeError('Only support torch.preserve_format for ShardedTensor!')
    cloned_local_shards = [Shard(local_shard.tensor.clone(memory_format=desire_memory_format), metadata=copy.deepcopy(local_shard.metadata)) for local_shard in self_st.local_shards()]
    new_metadata = copy.deepcopy(self_st.metadata())
    return (cloned_local_shards, new_metadata)