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
def class_hierarchy(cls):
    """Return an unordered sequence of all classes related to cls.

    Traverses diamond hierarchies.

    Fibs slightly: subclasses of builtin types are not returned.  Thus
    class_hierarchy(class A(object)) returns (A, object), not A plus every
    class systemwide that derives from object.

    """
    hier = {cls}
    process = list(cls.__mro__)
    while process:
        c = process.pop()
        bases = (_ for _ in c.__bases__ if _ not in hier)
        for b in bases:
            process.append(b)
            hier.add(b)
        if c.__module__ == 'builtins' or not hasattr(c, '__subclasses__'):
            continue
        for s in [_ for _ in (c.__subclasses__() if not issubclass(c, type) else c.__subclasses__(c)) if _ not in hier]:
            process.append(s)
            hier.add(s)
    return list(hier)