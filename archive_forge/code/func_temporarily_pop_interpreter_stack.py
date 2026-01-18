from abc import ABC, abstractmethod
import contextlib
from typing import Any
import torch
import torch.utils._pytree as pytree
from torch._C._functorch import (
from torch.autograd.forward_ad import _set_fwd_grad_enabled
@contextlib.contextmanager
def temporarily_pop_interpreter_stack():
    try:
        saved = pop_dynamic_layer_stack()
        yield
    finally:
        push_dynamic_layer_stack(saved)