import torch
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._shard.sharded_tensor._ops._common import _sharded_op_common
from torch.distributed._shard.sharding_spec import ChunkShardingSpec
from torch.distributed._shard.sharding_spec._internals import (
from torch.distributed._shard.sharding_spec.api import custom_sharding_spec_op
from torch.distributed.nn.functional import (
@custom_sharding_spec_op(ChunkShardingSpec, op)
@_sharded_op_common(op, early_stop_func, extra_check)
def sharded_tensor_op_on_local_tensor(types, args=(), kwargs=None, pg=None):
    st = args[0]
    sharding_spec = st.sharding_spec()
    if len(st.local_shards()) != 1:
        raise TypeError(f"torch function '{op.__name__}', with args: {args} and kwargs: {kwargs} only supported for single local tensor!")
    st_size = st.size()
    if customized_func:
        local_tensor, sharding_spec, st_size = customized_func(args, kwargs, pg)
    else:
        args = (st.local_tensor(), *args[1:])
        local_tensor = op(*args, **kwargs)
    return ShardedTensor._init_from_local_tensor(local_tensor.contiguous(), sharding_spec, st_size, process_group=pg, init_rrefs=st._init_rrefs)