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
def check_length(self, max_length: int) -> bool:
    """Check this line fits within a given number of cells."""
    start_length = len(self.whitespace) + cell_len(self.text) + cell_len(self.suffix)
    assert self.node is not None
    return self.node.check_length(start_length, max_length)