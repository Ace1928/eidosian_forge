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
def inject_docstring_text(given_doctext: Optional[str], injecttext: str, pos: int) -> str:
    doctext: str = _dedent_docstring(given_doctext or '')
    lines = doctext.split('\n')
    if len(lines) == 1:
        lines.append('')
    injectlines = textwrap.dedent(injecttext).split('\n')
    if injectlines[0]:
        injectlines.insert(0, '')
    blanks = [num for num, line in enumerate(lines) if not line.strip()]
    blanks.insert(0, 0)
    inject_pos = blanks[min(pos, len(blanks) - 1)]
    lines = lines[0:inject_pos] + injectlines + lines[inject_pos:]
    return '\n'.join(lines)