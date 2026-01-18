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
def functionalize_functorch_fn(interpreter, *args, **kwargs):
    return fn(_FunctorchFunctionalizeAPI(interpreter), *args, **kwargs)