import torch
import inspect
import numbers
import types
import typing
import enum
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, NamedTuple, cast, TYPE_CHECKING
from torch._jit_internal import boolean_dispatched
from ._compatibility import compatibility
from torch._ops import OpOverloadPacket, OpOverload
def is_homogeneous_tuple(t):
    if getattr(t, '__origin__', None) not in {tuple, Tuple}:
        return False
    contained = t.__args__
    if t.__args__ == ((),):
        return True
    return all((c is Ellipsis or issubclass(c, sig_el_type) for c in contained))