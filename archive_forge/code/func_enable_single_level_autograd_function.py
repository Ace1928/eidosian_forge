import contextlib
import torch
from torch._C._functorch import (
from typing import Union, Tuple
@contextlib.contextmanager
def enable_single_level_autograd_function():
    try:
        prev_state = get_single_level_autograd_function_allowed()
        set_single_level_autograd_function_allowed(True)
        yield
    finally:
        set_single_level_autograd_function_allowed(prev_state)