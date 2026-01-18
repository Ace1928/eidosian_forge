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
@contextlib.contextmanager
def temporarily_pop_mode(mode_stack):
    assert len(mode_stack) > 0
    top_mode = mode_stack.pop()
    try:
        yield top_mode
    finally:
        mode_stack.append(top_mode)