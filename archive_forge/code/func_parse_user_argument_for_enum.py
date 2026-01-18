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
def parse_user_argument_for_enum(arg: Any, choices: Dict[_E, List[Any]], name: str, resolve_symbol_names: bool=False) -> Optional[_E]:
    """Given a user parameter, parse the parameter into a chosen value
    from a list of choice objects, typically Enum values.

    The user argument can be a string name that matches the name of a
    symbol, or the symbol object itself, or any number of alternate choices
    such as True/False/ None etc.

    :param arg: the user argument.
    :param choices: dictionary of enum values to lists of possible
        entries for each.
    :param name: name of the argument.   Used in an :class:`.ArgumentError`
        that is raised if the parameter doesn't match any available argument.

    """
    for enum_value, choice in choices.items():
        if arg is enum_value:
            return enum_value
        elif resolve_symbol_names and arg == enum_value.name:
            return enum_value
        elif arg in choice:
            return enum_value
    if arg is None:
        return None
    raise exc.ArgumentError(f"Invalid value for '{name}': {arg!r}")