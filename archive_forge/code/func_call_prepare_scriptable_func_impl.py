import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import (
from torch.jit._recursive import (
from torch.jit._state import (
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location
def call_prepare_scriptable_func_impl(obj, memo):
    if not isinstance(obj, torch.nn.Module):
        return obj
    obj_id = id(obj)
    if obj_id in memo:
        return memo[id(obj)]
    obj = obj.__prepare_scriptable__() if hasattr(obj, '__prepare_scriptable__') else obj
    memo[obj_id] = obj
    new_obj_dict = {}
    for name, sub_module in obj.__dict__.items():
        if name == '_modules':
            for k, v in sub_module.items():
                sub_module[k] = call_prepare_scriptable_func_impl(v, memo)
            new_obj_dict[name] = sub_module
        elif isinstance(sub_module, torch.nn.Module) and (not isinstance(sub_module, ScriptModule)):
            new_obj_dict[name] = call_prepare_scriptable_func_impl(sub_module, memo)
        else:
            new_obj_dict[name] = sub_module
    for k, v in new_obj_dict.items():
        obj.__dict__[name] = v
    return obj