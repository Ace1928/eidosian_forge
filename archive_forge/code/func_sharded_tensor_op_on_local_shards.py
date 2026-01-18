import functools
from torch.distributed._shard.sharded_tensor import (
from torch.distributed._shard.common_op_utils import _basic_validation
@_sharded_op_impl(op)
@_sharded_op_common(op, early_stop_func, extra_check)
def sharded_tensor_op_on_local_shards(types, args=(), kwargs=None, pg=None):
    st = args[0]
    st_metadata = st.metadata()
    local_shards = st.local_shards()
    local_shards_new = []
    if customized_func:
        local_shards_new, st_metadata = customized_func(args, kwargs, pg)
    else:
        for local_shard in local_shards:
            args = (local_shard.tensor, *args[1:])
            local_shards_new.append(Shard(op(*args, **kwargs), local_shard.metadata))
    return ShardedTensor._init_from_local_shards_and_global_metadata(local_shards_new, st_metadata, process_group=pg, init_rrefs=st._init_rrefs, sharding_spec=st.sharding_spec())