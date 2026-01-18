import copy
import torch
from torch.distributed._shard.sharded_tensor import (
from ._common import (
from torch.distributed._shard.common_op_utils import _register_default_op
@_sharded_op_impl(torch.Tensor.is_meta.__get__)
def st_is_meta(types, args=(), kwargs=None, pg=None):
    return args[0].local_tensor().is_meta