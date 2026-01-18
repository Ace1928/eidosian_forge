from contextlib import contextmanager
import torch
import functools
from torch._decomp import decomposition_table
from typing import Callable, Dict
from torch.utils._pytree import tree_map_only
def implements_per_sample_grads(torch_function):

    @functools.wraps(torch_function)
    def decorator(autograd_func):
        HANDLED_FUNCTIONS[torch_function] = autograd_func
        return autograd_func
    return decorator