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
def quoted_token_parser(value):
    """Parse a dotted identifier with accommodation for quoted names.

    Includes support for SQL-style double quotes as a literal character.

    E.g.::

        >>> quoted_token_parser("name")
        ["name"]
        >>> quoted_token_parser("schema.name")
        ["schema", "name"]
        >>> quoted_token_parser('"Schema"."Name"')
        ['Schema', 'Name']
        >>> quoted_token_parser('"Schema"."Name""Foo"')
        ['Schema', 'Name""Foo']

    """
    if '"' not in value:
        return value.split('.')
    state = 0
    result: List[List[str]] = [[]]
    idx = 0
    lv = len(value)
    while idx < lv:
        char = value[idx]
        if char == '"':
            if state == 1 and idx < lv - 1 and (value[idx + 1] == '"'):
                result[-1].append('"')
                idx += 1
            else:
                state ^= 1
        elif char == '.' and state == 0:
            result.append([])
        else:
            result[-1].append(char)
        idx += 1
    return [''.join(token) for token in result]