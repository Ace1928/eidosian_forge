import os
import pathlib
import tempfile
import functools
import contextlib
import types
import importlib
import inspect
import warnings
import itertools
from typing import Union, Optional, cast
from .abc import ResourceReader, Traversable
from ._compat import wrap_spec
def package_to_anchor(func):
    """
    Replace 'package' parameter as 'anchor' and warn about the change.

    Other errors should fall through.

    >>> files('a', 'b')
    Traceback (most recent call last):
    TypeError: files() takes from 0 to 1 positional arguments but 2 were given
    """
    undefined = object()

    @functools.wraps(func)
    def wrapper(anchor=undefined, package=undefined):
        if package is not undefined:
            if anchor is not undefined:
                return func(anchor, package)
            warnings.warn("First parameter to files is renamed to 'anchor'", DeprecationWarning, stacklevel=2)
            return func(package)
        elif anchor is undefined:
            return func()
        return func(anchor)
    return wrapper