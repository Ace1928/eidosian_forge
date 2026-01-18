import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
def add_module_init_func(name: str, init_func: Callable[[], None]) -> None:
    """Register a module without eagerly importing it"""
    assert '.' not in name, f'Expected a root module name, but got {name}'
    if name in sys.modules:
        init_func()
    assert name not in _lazy_module_init
    _lazy_module_init[name].append(init_func)