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
def constructor_copy(obj: _T, cls: Type[_T], *args: Any, **kw: Any) -> _T:
    """Instantiate cls using the __dict__ of obj as constructor arguments.

    Uses inspect to match the named arguments of ``cls``.

    """
    names = get_cls_kwargs(cls)
    kw.update(((k, obj.__dict__[k]) for k in names.difference(kw) if k in obj.__dict__))
    return cls(*args, **kw)