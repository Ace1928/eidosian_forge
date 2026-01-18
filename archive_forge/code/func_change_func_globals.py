import ast
import inspect
import textwrap
import copy
import functools
from types import FunctionType
from typing import cast, Union, Callable, Dict, Optional, Any
from torch.fx._symbolic_trace import Tracer
from torch.fx.graph import Graph
from torch._sources import normalize_source_lines
import torch
def change_func_globals(f, globals):
    """Based on https://stackoverflow.com/a/13503277/2988730 (@unutbu)"""
    g = FunctionType(f.__code__, globals, name=f.__name__, argdefs=f.__defaults__, closure=f.__closure__)
    g = functools.update_wrapper(g, f)
    g.__kwdefaults__ = copy.copy(f.__kwdefaults__)
    return g