import functools
import importlib
import sys
import types
import torch
from .allowed_functions import _disallowed_function_ids, is_user_defined_allowed
from .utils import hashable
from .variables import (
@functools.lru_cache(None)
def get_torch_obj_rule_map():
    d = dict()
    for k, v in torch_name_rule_map.items():
        obj = load_object(k)
        if obj is not None:
            d[obj] = v
    return d