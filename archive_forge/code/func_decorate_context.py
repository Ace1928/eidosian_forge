import functools
import inspect
import warnings
import sys
from typing import Any, Callable, TypeVar, cast
@functools.wraps(func)
def decorate_context(*args, **kwargs):
    with ctx_factory():
        return func(*args, **kwargs)