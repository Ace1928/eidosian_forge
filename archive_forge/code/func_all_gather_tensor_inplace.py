import warnings
import sys
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import Tuple, Union, List, Optional, cast, TYPE_CHECKING
from . import _functional_collectives_impl as fun_col_impl
from ._functional_collectives_impl import _register_tensor_wrapper
from torch.fx.experimental.proxy_tensor import (
from torch._custom_ops import impl_abstract
from torch.distributed.distributed_c10d import (
def all_gather_tensor_inplace(output: torch.Tensor, input: torch.Tensor, group, async_op: bool=False, tag: str='', gather_dim: int=0):
    assert not async_op, "Can't remap async version of inplace op to functional collective"
    return output.copy_(all_gather_tensor(input, gather_dim, group, tag))