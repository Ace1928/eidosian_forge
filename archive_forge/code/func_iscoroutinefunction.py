import collections.abc
import inspect
import os
import sys
import traceback
import types
def iscoroutinefunction(func):
    """Return True if func is a decorated coroutine function."""
    return inspect.iscoroutinefunction(func) or getattr(func, '_is_coroutine', None) is _is_coroutine