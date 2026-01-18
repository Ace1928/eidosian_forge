import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
def ensure_lazy(array):
    if not isinstance(array, LazyArray):
        return LazyArray.from_data(array)
    return array