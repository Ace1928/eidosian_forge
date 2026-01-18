import functools
from typing import List
import torch
import torch.distributed._shard.sharding_spec as shard_spec
from .api import (
from .metadata import ShardMetadata  # noqa: F401
from torch.distributed._shard.op_registry_utils import _decorator_func
from ._ops import *  # noqa: F403
def init_from_local_shards(local_shards: List[Shard], *global_size, process_group=None, init_rrefs=False) -> ShardedTensor:
    """
    Creates an :class:`ShardedTensor` from local shards and the global metadata.
    Needs to be called on all ranks in an SPMD fashion.

    Args:
        local_shards (List[:class `torch.distributed._shard.sharded_tensor.Shard`]): A list
            of shards that represent the local shards on this rank.
        global_size (int...):  a list, tuple, or `torch.Size` of integers defining the
            shape of the overall sharded tensor.

    Keyword args:
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object handle on this rank


    Examples:
        Suppose we want construct a sharded tensor on two ranks, global size = (10, 5),
        each shard have a (5, 5) local tensor, we can do it like below:

        on rank 0:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[0, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:0/cuda:0"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])

        on rank 1:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[5, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:1/cuda:1"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])
    """
    return ShardedTensor._init_from_local_shards(local_shards, *global_size, process_group=process_group, init_rrefs=init_rrefs)