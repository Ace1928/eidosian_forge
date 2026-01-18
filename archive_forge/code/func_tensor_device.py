import copy
import torch
from torch.distributed._shard.sharded_tensor import (
from ._common import (
from torch.distributed._shard.common_op_utils import _register_default_op
@_sharded_op_impl(torch.Tensor.device.__get__)
def tensor_device(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    if not isinstance(self_st, ShardedTensor):
        raise TypeError('input needs to be a ShardedTensor')
    dev: torch.device
    if self_st._local_shards:
        dev = self_st._local_shards[0].tensor.device
    elif pg and pg._get_backend_name() == 'gloo':
        dev = torch.device('cpu')
    else:
        dev = torch.device(torch.cuda.current_device())
    return dev