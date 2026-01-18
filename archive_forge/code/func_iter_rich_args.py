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
def iter_rich_args(rich_args: Any) -> Iterable[Union[Any, Tuple[str, Any]]]:
    for arg in rich_args:
        if _safe_isinstance(arg, tuple):
            if len(arg) == 3:
                key, child, default = arg
                if default == child:
                    continue
                yield (key, child)
            elif len(arg) == 2:
                key, child = arg
                yield (key, child)
            elif len(arg) == 1:
                yield arg[0]
        else:
            yield arg