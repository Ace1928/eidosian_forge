import sys
from functools import wraps
from types import coroutine
import inspect
from inspect import (
import collections.abc
@wraps(coroutine_maker)
def async_generator_maker(*args, **kwargs):
    return AsyncGenerator(coroutine_maker(*args, **kwargs))