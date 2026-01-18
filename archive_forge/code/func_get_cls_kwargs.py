from __future__ import annotations
import collections
import enum
from functools import update_wrapper
import inspect
import itertools
import operator
import re
import sys
import textwrap
import threading
import types
from types import CodeType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
from . import _collections
from . import compat
from ._has_cy import HAS_CYEXTENSION
from .typing import Literal
from .. import exc
def get_cls_kwargs(cls: type, *, _set: Optional[Set[str]]=None, raiseerr: bool=False) -> Optional[Set[str]]:
    """Return the full set of inherited kwargs for the given `cls`.

    Probes a class's __init__ method, collecting all named arguments.  If the
    __init__ defines a \\**kwargs catch-all, then the constructor is presumed
    to pass along unrecognized keywords to its base classes, and the
    collection process is repeated recursively on each of the bases.

    Uses a subset of inspect.getfullargspec() to cut down on method overhead,
    as this is used within the Core typing system to create copies of type
    objects which is a performance-sensitive operation.

    No anonymous tuple arguments please !

    """
    toplevel = _set is None
    if toplevel:
        _set = set()
    assert _set is not None
    ctr = cls.__dict__.get('__init__', False)
    has_init = ctr and isinstance(ctr, types.FunctionType) and isinstance(ctr.__code__, types.CodeType)
    if has_init:
        names, has_kw = _inspect_func_args(ctr)
        _set.update(names)
        if not has_kw and (not toplevel):
            if raiseerr:
                raise TypeError(f"given cls {cls} doesn't have an __init__ method")
            else:
                return None
    else:
        has_kw = False
    if not has_init or has_kw:
        for c in cls.__bases__:
            if get_cls_kwargs(c, _set=_set) is None:
                break
    _set.discard('self')
    return _set