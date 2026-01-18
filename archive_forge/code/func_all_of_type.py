import typing
import types
import inspect
import functools
from . import _uarray
import copyreg
import pickle
import contextlib
from ._uarray import (  # type: ignore
def all_of_type(arg_type):
    """
    Marks all unmarked arguments as a given type.

    Examples
    --------
    >>> @all_of_type(str)
    ... def f(a, b):
    ...     return a, Dispatchable(b, int)
    >>> f('a', 1)
    (<Dispatchable: type=<class 'str'>, value='a'>,
     <Dispatchable: type=<class 'int'>, value=1>)
    """

    def outer(func):

        @functools.wraps(func)
        def inner(*args, **kwargs):
            extracted_args = func(*args, **kwargs)
            return tuple((Dispatchable(arg, arg_type) if not isinstance(arg, Dispatchable) else arg for arg in extracted_args))
        return inner
    return outer