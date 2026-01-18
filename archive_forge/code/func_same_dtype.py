import copy
import torch
from torch.distributed._shard.sharded_tensor import (
from ._common import (
from torch.distributed._shard.common_op_utils import _register_default_op
def same_dtype(*args, **kwargs):
    """
    When the dtype is the same, return the original ShardedTensor.

    Args: same as ``torch.Tensor.type_as``.

    Return (bool): Whether to return early or not.
    """
    return args[0].dtype == args[1].dtype