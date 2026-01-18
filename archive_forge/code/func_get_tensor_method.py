import functools
import importlib
import sys
import types
import torch
from .allowed_functions import _disallowed_function_ids, is_user_defined_allowed
from .utils import hashable
from .variables import (
@functools.lru_cache(None)
def get_tensor_method():
    s = set()
    for name in dir(torch.Tensor):
        method = getattr(torch.Tensor, name)
        if isinstance(method, (types.MethodDescriptorType, types.WrapperDescriptorType)):
            s.add(method)
    return frozenset(s)