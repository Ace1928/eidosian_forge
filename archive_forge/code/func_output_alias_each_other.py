import functools
import warnings
from typing import Callable, Union
import torch
import torch.utils._pytree as pytree
from torch._ops import OpOverload
from torch._subclasses.fake_tensor import (
from torch.utils._python_dispatch import TorchDispatchMode
def output_alias_each_other(outputs):
    storages = set()
    for out in tree_flatten_only(torch.Tensor, outputs):
        if not torch._C._has_storage(out):
            continue
        stor = out._typed_storage()._cdata
        if stor in storages:
            return True
        storages.add(stor)
    return False