import functools
import inspect
import textwrap
import threading
import types
import warnings
from typing import TypeVar, Type, Callable, Any, Union
from inspect import signature
from functools import wraps
def deprecate_function(func, message, warning_type=warning_type):
    """
        Returns a wrapped function that displays ``warning_type``
        when it is called.
        """
    if isinstance(func, method_types):
        func_wrapper = type(func)
    else:
        func_wrapper = lambda f: f
    func = get_function(func)

    def deprecated_func(*args, **kwargs):
        if pending:
            category = PendingDeprecationWarning
        else:
            category = warning_type
        warnings.warn(message, category, stacklevel=2)
        return func(*args, **kwargs)
    if type(func) is not type(str.__dict__['__add__']):
        deprecated_func = functools.wraps(func)(deprecated_func)
    deprecated_func.__doc__ = deprecate_doc(deprecated_func.__doc__, message)
    return func_wrapper(deprecated_func)