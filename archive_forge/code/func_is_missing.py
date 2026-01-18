import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def is_missing(modname, name, func):
    if name.startswith('_') and (not name.startswith('__')):
        return False
    if name.startswith('__pyx_unpickle_') or name.endswith('_cython__'):
        return False
    try:
        if issubclass(func, BaseException):
            return False
    except TypeError:
        pass
    try:
        return callable(func) and func.__module__ is not None and (modname in func.__module__) and (is_partial_args(func, (), {}) is not True) and (func not in denylist)
    except AttributeError:
        return False