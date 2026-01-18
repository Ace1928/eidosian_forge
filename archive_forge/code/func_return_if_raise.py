from __future__ import annotations
import cProfile
import pstats
import signal
import sys
import tempfile
from collections import namedtuple
from functools import partial, wraps
from typing import Any, Callable, Union
def return_if_raise(exception_tuple: Union[list, tuple], retval_if_exc: Any, disabled: bool=False) -> Any:
    """
    Decorator for functions, methods or properties. Execute the callable in a
    try block, and return retval_if_exc if one of the exceptions listed in
    exception_tuple is raised (se also ``return_node_if_raise``).

    Setting disabled to True disables the try except block (useful for
    debugging purposes). One can use this decorator to define properties.

    Examples:
        @return_if_raise(ValueError, None)
        def return_none_if_value_error(self):
            pass

        @return_if_raise((ValueError, KeyError), "hello")
        def another_method(self):
            pass

        @property
        @return_if_raise(AttributeError, None)
        def name(self):
            "Name of the object, None if not set."
            return self._name

    """
    if isinstance(exception_tuple, list):
        exception_tuple = tuple(exception_tuple)
    elif not isinstance(exception_tuple, tuple):
        exception_tuple = (exception_tuple,)
    else:
        raise TypeError(f'Wrong exception_tuple {type(exception_tuple)}')

    def decorator(func):

        @wraps(func)
        def wrapper(*args, **kwargs):
            if disabled:
                return func(*args, **kwargs)
            try:
                return func(*args, **kwargs)
            except exception_tuple:
                return retval_if_exc
        return wrapper
    return decorator