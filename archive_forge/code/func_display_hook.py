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
def display_hook(value: Any) -> None:
    """Replacement sys.displayhook which prettifies objects with Rich."""
    if value is not None:
        assert console is not None
        builtins._ = None
        console.print(value if _safe_isinstance(value, RichRenderable) else Pretty(value, overflow=overflow, indent_guides=indent_guides, max_length=max_length, max_string=max_string, max_depth=max_depth, expand_all=expand_all), crop=crop)
        builtins._ = value