import contextlib
import functools
import warnings
from typing import Callable, Optional
import torch
from torch._library.utils import Kernel, RegistrationHandle
@contextlib.contextmanager
def set_ctx_getter(ctx_getter):
    global global_ctx_getter
    prev = global_ctx_getter
    try:
        global_ctx_getter = ctx_getter
        yield
    finally:
        global_ctx_getter = prev