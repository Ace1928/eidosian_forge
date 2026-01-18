import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import (
import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager
def replace_ph(x):
    nonlocal cnt
    cnt += 1
    param = sig.parameters[name]
    default = () if param.default is inspect.Parameter.empty else (param.default,)
    out = self.create_proxy('placeholder', f'{name}_{str(cnt)}', default, {})
    if isinstance(x, PHBase):

        def transfer_attrs(fr, to):
            for attr_name in dir(fr):
                attr_val = getattr(fr, attr_name)
                if not callable(attr_val) and (not attr_name.startswith('__')) and (not hasattr(to, attr_name)):
                    setattr(to, attr_name, attr_val)
        if x != PH:
            transfer_attrs(fr=x, to=out.node)
        return out
    if type(x) == bool or (type(x) in base_types and type(x) != torch.Tensor):
        torch._assert(out == x, f'{name} has been specialized to have value {x} but got another value')
    elif type(x) == type(None):
        args = (out, f'{name} has been specialized to have value None but got another value')
        self.create_proxy('call_function', _assert_is_none, args, {})
    else:
        warnings.warn(f'Was not able to add assertion to guarantee correct input {name} to specialized function. It is up to the user to make sure that your inputs match the inputs you specialized the function with.')
    return x