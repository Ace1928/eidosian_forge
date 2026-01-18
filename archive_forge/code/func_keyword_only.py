import inspect
import re
import types
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, Union
def keyword_only(func):
    """A decorator that forces keyword arguments in the wrapped method."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            raise TypeError(f'Method {func.__name__} only takes keyword arguments.')
        return func(**kwargs)
    indent = _get_min_indent_of_docstring(wrapper.__doc__)
    notice = indent + '.. note:: This method requires all argument be specified by keyword.\n'
    wrapper.__doc__ = notice + wrapper.__doc__ if wrapper.__doc__ else notice
    return wrapper