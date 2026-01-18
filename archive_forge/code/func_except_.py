import functools
import time
import inspect
import collections
import types
import itertools
import warnings
import setuptools.extern.more_itertools
from typing import Callable, TypeVar
def except_(*exceptions, replace=None, use=None):
    """
    Replace the indicated exceptions, if raised, with the indicated
    literal replacement or evaluated expression (if present).

    >>> safe_int = except_(ValueError)(int)
    >>> safe_int('five')
    >>> safe_int('5')
    5

    Specify a literal replacement with ``replace``.

    >>> safe_int_r = except_(ValueError, replace=0)(int)
    >>> safe_int_r('five')
    0

    Provide an expression to ``use`` to pass through particular parameters.

    >>> safe_int_pt = except_(ValueError, use='args[0]')(int)
    >>> safe_int_pt('five')
    'five'

    """

    def decorate(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exceptions:
                try:
                    return eval(use)
                except TypeError:
                    return replace
        return wrapper
    return decorate