import contextlib
from abc import ABC, abstractmethod
from typing import Any, Callable, ContextManager, Tuple
import torch
import torch.utils._pytree as pytree
from torch._C import _functionalization_reapply_views_tls as _reapply_views
from torch.utils._python_dispatch import return_and_correct_aliasing, TorchDispatchMode
@contextlib.contextmanager
def maybe_disable_functional_mode():
    maybe_func_mode = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FUNCTIONAL)
    try:
        yield
    finally:
        if maybe_func_mode is not None:
            torch._C._set_dispatch_mode(maybe_func_mode)