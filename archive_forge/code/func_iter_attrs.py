import builtins
import collections
import dataclasses
import inspect
import os
import sys
from array import array
from collections import Counter, UserDict, UserList, defaultdict, deque
from dataclasses import dataclass, fields, is_dataclass
from inspect import isclass
from itertools import islice
from types import MappingProxyType
from typing import (
from pip._vendor.rich.repr import RichReprResult
from . import get_console
from ._loop import loop_last
from ._pick import pick_bool
from .abc import RichRenderable
from .cells import cell_len
from .highlighter import ReprHighlighter
from .jupyter import JupyterMixin, JupyterRenderable
from .measure import Measurement
from .text import Text
def iter_attrs() -> Iterable[Tuple[str, Any, Optional[Callable[[Any], str]]]]:
    """Iterate over attr fields and values."""
    for attr in attr_fields:
        if attr.repr:
            try:
                value = getattr(obj, attr.name)
            except Exception as error:
                yield (attr.name, error, None)
            else:
                yield (attr.name, value, attr.repr if callable(attr.repr) else None)