import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@property
def fn_name(self):
    """The name of the function to use to compute this array."""
    return getattr(self._fn, '__name__', 'None')