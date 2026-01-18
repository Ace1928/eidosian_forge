import contextlib
import functools
import warnings
from typing import Callable, Optional
import torch
from torch._library.utils import Kernel, RegistrationHandle
def error_on_ctx():
    raise RuntimeError(f'Attempted to call get_ctx() for the meta implementation for {qualname} (implemented at {source})You have presumably called get_ctx() because the operator has a data-dependent output shape; if so, there is no such meta implementation and this error is the correct behavior.')