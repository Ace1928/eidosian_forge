import copy
import torch
from torch.distributed._shard.sharded_tensor import (
from ._common import (
from torch.distributed._shard.common_op_utils import _register_default_op
@_sharded_op_impl(torch.Tensor.requires_grad_)
def tensor_requires_grad_set(types, args=(), kwargs=None, pg=None):
    self_st = args[0]
    if not isinstance(self_st, ShardedTensor):
        raise TypeError('input needs to be a ShardedTensor')
    if kwargs is None:
        kwargs = {}
    requires_grad = args[1] if len(args) > 1 else kwargs.get('requires_grad', True)
    if requires_grad == self_st.requires_grad:
        return self_st
    for local_shard in self_st.local_shards():
        local_shard.tensor.requires_grad_(requires_grad)
    with torch._C.DisableTorchFunctionSubclass():
        self_st.requires_grad_(requires_grad)
    self_st._metadata.tensor_properties.requires_grad = requires_grad
    return self_st