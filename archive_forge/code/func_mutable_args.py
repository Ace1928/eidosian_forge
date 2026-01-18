import weakref
import torch
import torch.utils._pytree as pytree
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._ops import OpOverload
from torch.library import Library
from torchgen.model import (
from .autograd import autograd_not_implemented
def mutable_args(op: OpOverload):
    return tuple((False if arg.alias_info is None else arg.alias_info.is_write for arg in op._schema.arguments))