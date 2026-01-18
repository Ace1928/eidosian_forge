from typing import TYPE_CHECKING, Union, Callable, Any, Tuple, List, Optional, Dict, Set
from ._compatibility import compatibility
from .immutable_collections import immutable_dict, immutable_list
import torch
import builtins
import types
import inspect
import warnings
from torch.fx.operator_schemas import normalize_function, normalize_module, ArgsKwargsPair
from .._ops import ops as _ops
@compatibility(is_backward_compatible=True)
def map_aggregate(a: Argument, fn: Callable[[Argument], Argument]) -> Argument:
    """
    Apply fn to each Node appearing arg. arg may be a list, tuple, slice, or dict with string keys.
    """
    if isinstance(a, tuple):
        t = tuple((map_aggregate(elem, fn) for elem in a))
        return t if not hasattr(a, '_fields') else type(a)(*t)
    elif isinstance(a, list):
        return immutable_list((map_aggregate(elem, fn) for elem in a))
    elif isinstance(a, dict):
        return immutable_dict(((k, map_aggregate(v, fn)) for k, v in a.items()))
    elif isinstance(a, slice):
        return slice(map_aggregate(a.start, fn), map_aggregate(a.stop, fn), map_aggregate(a.step, fn))
    else:
        return fn(a)