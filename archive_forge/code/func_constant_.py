import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharded_tensor import (
@_sharded_op_impl(torch.nn.init.constant_)
def constant_(types, args=(), kwargs=None, pg=None):
    """
    Fills the input ShardedTensor with the value \\text{val}val.
    Args:
        tensor: tensor sharded across devices
        val: the value to fill the tensor with
    """
    validate_param(kwargs, 'kwargs')
    sharded_tensor = kwargs['tensor']
    validate_param(sharded_tensor, 'tensor')
    val = kwargs['val']
    validate_param(val, 'val')
    for shard in sharded_tensor.local_shards():
        torch.nn.init.constant_(shard.tensor, val=val)
    return sharded_tensor