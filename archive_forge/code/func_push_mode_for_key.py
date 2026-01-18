import contextlib
import ctypes
import importlib
import inspect
import sys
import types
from typing import Any, Callable, Dict, List, Type, Union
import torch._C
import torch.utils._pytree as pytree
from torch import _utils_internal
from torch._functorch.pyfunctorch import dispatch_functorch
def push_mode_for_key(key, mode):
    assert isinstance(key, torch._C.DispatchKey)
    assert isinstance(mode, torch.utils._python_dispatch.TorchDispatchMode)
    if key not in mode_stack_per_key():
        mode_stack_per_key()[key] = []
    mode_stack_per_key()[key].append(mode)