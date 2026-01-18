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
@functools.wraps(_orig_module_call)
def module_call_wrapper(mod, *args, **kwargs):

    def forward(*args, **kwargs):
        return _orig_module_call(mod, *args, **kwargs)
    _autowrap_check(patcher, getattr(getattr(mod, 'forward', mod), '__globals__', {}), self._autowrap_function_ids)
    return self.call_module(mod, forward, args, kwargs)