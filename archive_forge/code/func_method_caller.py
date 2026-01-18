import functools
import time
import inspect
import collections
import types
import itertools
import warnings
import setuptools.extern.more_itertools
from typing import Callable, TypeVar
def method_caller(method_name, *args, **kwargs):
    """
    Return a function that will call a named method on the
    target object with optional positional and keyword
    arguments.

    >>> lower = method_caller('lower')
    >>> lower('MyString')
    'mystring'
    """

    def call_method(target):
        func = getattr(target, method_name)
        return func(*args, **kwargs)
    return call_method