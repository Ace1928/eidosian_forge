from abc import abstractmethod, ABCMeta
import collections
from collections import defaultdict
import collections.abc
import contextlib
import functools
import operator
import re as stdlib_re  # Avoid confusion with the re we export.
import sys
import types
import warnings
from types import WrapperDescriptorType, MethodWrapperType, MethodDescriptorType, GenericAlias
def no_type_check(arg):
    """Decorator to indicate that annotations are not type hints.

    The argument must be a class or function; if it is a class, it
    applies recursively to all methods and classes defined in that class
    (but not to methods defined in its superclasses or subclasses).

    This mutates the function(s) or class(es) in place.
    """
    if isinstance(arg, type):
        for key in dir(arg):
            obj = getattr(arg, key)
            if not hasattr(obj, '__qualname__') or obj.__qualname__ != f'{arg.__qualname__}.{obj.__name__}' or getattr(obj, '__module__', None) != arg.__module__:
                continue
            if isinstance(obj, types.FunctionType):
                obj.__no_type_check__ = True
            if isinstance(obj, types.MethodType):
                obj.__func__.__no_type_check__ = True
            if isinstance(obj, type):
                no_type_check(obj)
    try:
        arg.__no_type_check__ = True
    except TypeError:
        pass
    return arg