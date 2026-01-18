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
def coerce_kw_type(kw: Dict[str, Any], key: str, type_: Type[Any], flexi_bool: bool=True, dest: Optional[Dict[str, Any]]=None) -> None:
    """If 'key' is present in dict 'kw', coerce its value to type 'type\\_' if
    necessary.  If 'flexi_bool' is True, the string '0' is considered false
    when coercing to boolean.
    """
    if dest is None:
        dest = kw
    if key in kw and (not isinstance(type_, type) or not isinstance(kw[key], type_)) and (kw[key] is not None):
        if type_ is bool and flexi_bool:
            dest[key] = asbool(kw[key])
        else:
            dest[key] = type_(kw[key])