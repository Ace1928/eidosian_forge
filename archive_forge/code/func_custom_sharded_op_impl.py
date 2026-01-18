import functools
from typing import List
import torch
import torch.distributed._shard.sharding_spec as shard_spec
from .api import (
from .metadata import ShardMetadata  # noqa: F401
from torch.distributed._shard.op_registry_utils import _decorator_func
from ._ops import *  # noqa: F403
def custom_sharded_op_impl(func):
    """
    Provides a way for users to write their own custom sharded operator. This
    can be used to override existing ShardedTensor operators or write a new
    one not supported by ShardedTensor. If the operator in question is covered
    by ``__torch_function__`` dispatch and has a ShardedTensor as any of its
    parameters, the function provided will be invoked for that operator.

    Example::
        >>> # xdoctest: +SKIP
        >>> @custom_sharded_op_impl(torch.nn.functional.linear)
        >>> def my_custom_sharded_linear(types, args, kwargs, process_group):
        >>>     ...
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> input = torch.rand(10, 32)
        >>> weight = sharded_tensor.rand(32, 16)
        >>> bias = torch.rand(16)
        >>> # This will call 'my_custom_sharded_linear'
        >>> torch.nn.functional.linear(input, weight, bias)

    The types, args and kwargs parameters are the same parameters that are
    passed to ``__torch_function__`` dispatch API
    (https://pytorch.org/docs/stable/notes/extending.html#extending-torch).
    There is an additional ``process_group`` parameter which is the
    process_group used for the ShardedTensor and can be used by
    implementations for communications within a sharded implementation.

    Args:
        func(Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.nn.functional.linear)
    """
    return functools.partial(_decorator_func, op=func, op_table=_CUSTOM_SHARDED_OPS)