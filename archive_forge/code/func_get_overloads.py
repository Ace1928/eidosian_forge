import abc
import collections
import collections.abc
import functools
import inspect
import operator
import sys
import types as _types
import typing
import warnings
def get_overloads(func):
    """Return all defined overloads for *func* as a sequence."""
    f = getattr(func, '__func__', func)
    if f.__module__ not in _overload_registry:
        return []
    mod_dict = _overload_registry[f.__module__]
    if f.__qualname__ not in mod_dict:
        return []
    return list(mod_dict[f.__qualname__].values())