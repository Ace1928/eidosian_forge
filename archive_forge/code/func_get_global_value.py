import dis
from contextlib import contextmanager
import builtins
import operator
from typing import Iterator
from functools import reduce
from numba.core import (
from numba.core.utils import (
from .rvsdg.bc2rvsdg import (
from .rvsdg.regionpasses import RegionVisitor
def get_global_value(self, name):
    """THIS IS COPIED from interpreter.py

        Get a global value from the func_global (first) or
        as a builtins (second).  If both failed, return a ir.UNDEFINED.
        """
    try:
        return self.func_id.func.__globals__[name]
    except KeyError:
        return getattr(builtins, name, ir.UNDEFINED)