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
def is_allowed(obj) -> bool:
    """Is this safe to trace like torch.add ?"""
    _maybe_init_lazy_module(obj)
    if id(obj) in _disallowed_function_ids:
        return False
    if id(obj) in _allowed_function_ids:
        return True
    return isinstance(obj, (torch._ops.OpOverloadPacket, torch._ops.OpOverload, torch._ops._OpNamespace))