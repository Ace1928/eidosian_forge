import inspect
import warnings
import types
import collections
import itertools
from functools import lru_cache, wraps
from typing import Callable, List, Union, Iterable, TypeVar, cast
def is_consecutive(c):
    c_int = ord(c)
    is_consecutive.prev, prev = (c_int, is_consecutive.prev)
    if c_int - prev > 1:
        is_consecutive.value = next(is_consecutive.counter)
    return is_consecutive.value