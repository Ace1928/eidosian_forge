import contextlib
import threading
from contextvars import ContextVar
from typing import Any, Callable, Dict, Optional, Type, cast
from .. import registry
from ..compat import cupy, has_cupy
from ..util import (
from ._cupy_allocators import cupy_pytorch_allocator, cupy_tensorflow_allocator
from ._param_server import ParamServer
from .cupy_ops import CupyOps
from .mps_ops import MPSOps
from .numpy_ops import NumpyOps
from .ops import Ops
def set_gpu_allocator(allocator: str) -> None:
    """Route GPU memory allocation via PyTorch or tensorflow.
    Raise an error if the given argument does not match either of the two.
    """
    if allocator == 'pytorch':
        use_pytorch_for_gpu_memory()
    elif allocator == 'tensorflow':
        use_tensorflow_for_gpu_memory()
    else:
        raise ValueError(f"Invalid 'gpu_allocator' argument: '{allocator}'. Available allocators are: 'pytorch', 'tensorflow'")