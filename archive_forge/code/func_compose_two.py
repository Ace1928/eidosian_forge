import functools
import time
import inspect
import collections
import types
import itertools
import warnings
import setuptools.extern.more_itertools
from typing import Callable, TypeVar
def compose_two(f1, f2):
    return lambda *args, **kwargs: f1(f2(*args, **kwargs))