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
def expand_unions(type_: Type[Any], include_union: bool=False, discard_none: bool=False) -> Tuple[Type[Any], ...]:
    """Return a type as a tuple of individual types, expanding for
    ``Union`` types."""
    if is_union(type_):
        typ = set(type_.__args__)
        if discard_none:
            typ.discard(NoneType)
        if include_union:
            return (type_,) + tuple(typ)
        else:
            return tuple(typ)
    else:
        return (type_,)