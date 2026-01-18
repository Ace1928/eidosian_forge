from contextlib import contextmanager
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed import distributed_c10d
from torch.distributed._shard.sharded_tensor import (
from .sharding_spec import (
from .sharding_plan import (
from .sharder import Sharder
def shard_parameter(module: torch.nn.Module, param_name: str, sharding_spec: ShardingSpec, src_rank=0, process_group=None):
    """
    Given a :class:`torch.nn.Module`, a ``param_name`` for a parameter in that
    module, it shards that parameter according to the provided
    ``sharding_spec``. ``src_rank`` denotes the source rank which would be
    used as the ground truth of the data which would be scattered as shards
    across the rest of the ranks.

    This method replaces ``module.param_name`` with a
    :class:`torch.distributed._sharded_tensor.ShardedTensor`

    Args:
        module (:class:`torch.nn.Module`): Module whose parameter needs to be sharded.
        param_name (str): Name of the parameter of ``module`` that needs to be sharded.
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.

    Keyword args:
        src_rank (int, optional): The source rank which is used as the ground truth of
            the data for the parameter that would be sharded and scattered
            across the rest of the ranks.
            Default: 0.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    .. warning::
        Only :class:`torch.distributed._shard.sharding_spec.ChunkShardingSpec` is
        currently supported as the ``sharding_spec``.
    """
    if not hasattr(module, param_name):
        raise AttributeError(f'{module._get_name()} has no attribute `{param_name}`')
    tensor = getattr(module, param_name)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(f'Expected {type(module).__name__}.{param_name} to be a Tensor, but found {type(tensor).__name__}')
    if not tensor.is_contiguous():
        raise ValueError(f'param: {param_name} is not a contiguous Tensor')
    st = _shard_tensor(tensor, sharding_spec, src_rank, process_group)
    module.register_parameter(param_name, nn.Parameter(st))