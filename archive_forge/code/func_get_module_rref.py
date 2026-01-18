import collections
import io
import sys
import types
from typing import (
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
def get_module_rref(self) -> rpc.RRef[nn.Module]:
    """Return an :class:`~torch.distributed.rpc.RRef` (``RRef[nn.Module]``) pointing to the remote module."""
    return self.module_rref