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
def repr_tuple_names(names: List[str]) -> Optional[str]:
    """Trims a list of strings from the middle and return a string of up to
    four elements. Strings greater than 11 characters will be truncated"""
    if len(names) == 0:
        return None
    flag = len(names) <= 4
    names = names[0:4] if flag else names[0:3] + names[-1:]
    res = ['%s..' % name[:11] if len(name) > 11 else name for name in names]
    if flag:
        return ', '.join(res)
    else:
        return '%s, ..., %s' % (', '.join(res[0:3]), res[-1])