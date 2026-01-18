import copy
import torch
from torch.distributed._shard.sharded_tensor import (
from ._common import (
from torch.distributed._shard.common_op_utils import _register_default_op
def sharded_type_as_check(*args, **kwargs):
    """
    Perform extra checks for the sharded_type_as op such as the input needs to
    be either a Tensor or ShardedTensor.

    Args: same as ``torch.Tensor.type_as``.

    Return: None
    """
    if len(args) < 2:
        raise ValueError('Needs to give a tensor to cast type as!')
    if not isinstance(args[1], torch.Tensor) and (not isinstance(args[1], ShardedTensor)):
        raise ValueError('Needs to give a Tensor or ShardedTensor to cast type as!')