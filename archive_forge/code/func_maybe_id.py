import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def maybe_id(x):
    if hasattr(x, 'shape'):
        return id(x)
    return x