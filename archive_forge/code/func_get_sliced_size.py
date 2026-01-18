import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@functools.lru_cache(2 ** 12)
def get_sliced_size(d, start, stop, step):
    return len(range(d)[slice(start, stop, step)])