import functools
import inspect
import warnings
from collections import OrderedDict
from typing import Any, List, Optional, Tuple
import torch
import torch._C as _C
import torch._functorch as _functorch
import torch.utils.hooks as hooks
from torch._C import _functions
from torch._functorch.autograd_function import custom_function_call
def mark_shared_storage(self, *pairs):
    warnings.warn('mark_shared_storage is deprecated. Tensors with shared storages are automatically tracked. Note that calls to `set_()` are not tracked')