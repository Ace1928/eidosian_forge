from __future__ import annotations
import builtins
import collections.abc as collections_abc
import re
import sys
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import ForwardRef
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from . import compat
def is_origin_of_cls(type_: Any, class_obj: Union[Tuple[Type[Any], ...], Type[Any]]) -> bool:
    """return True if the given type has an __origin__ that shares a base
    with the given class"""
    origin = typing_get_origin(type_)
    if origin is None:
        return False
    return isinstance(origin, type) and issubclass(origin, class_obj)