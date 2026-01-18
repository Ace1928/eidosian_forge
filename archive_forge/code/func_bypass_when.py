import collections.abc
import functools
import inspect
import itertools
import operator
import time
import types
import warnings
import more_itertools
def bypass_when(check, *, _op=identity):
    """
    Decorate a function to return its parameter when ``check``.

    >>> bypassed = []  # False

    >>> @bypass_when(bypassed)
    ... def double(x):
    ...     return x * 2
    >>> double(2)
    4
    >>> bypassed[:] = [object()]  # True
    >>> double(2)
    2
    """

    def decorate(func):

        @functools.wraps(func)
        def wrapper(param, /):
            return param if _op(check) else func(param)
        return wrapper
    return decorate