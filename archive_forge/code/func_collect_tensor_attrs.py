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
def collect_tensor_attrs(m: torch.nn.Module, prefix_atoms: List[str]):
    for k, v in m.__dict__.items():
        if isinstance(v, (torch.Tensor, ScriptObject)):
            self.tensor_attrs[v] = '.'.join(prefix_atoms + [k])
    for k, v in m.named_children():
        collect_tensor_attrs(v, prefix_atoms + [k])