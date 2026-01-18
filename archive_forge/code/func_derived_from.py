from __future__ import annotations
import codecs
import functools
import inspect
import os
import re
import shutil
import sys
import tempfile
import types
import uuid
import warnings
from collections.abc import Hashable, Iterable, Iterator, Mapping, Set
from contextlib import contextmanager, nullcontext, suppress
from datetime import datetime, timedelta
from errno import ENOENT
from functools import lru_cache, wraps
from importlib import import_module
from numbers import Integral, Number
from operator import add
from threading import Lock
from typing import Any, Callable, ClassVar, Literal, TypeVar, cast, overload
from weakref import WeakValueDictionary
import tlz as toolz
from dask import config
from dask.core import get_deps
from dask.typing import no_default
def derived_from(original_klass, version=None, ua_args=None, skipblocks=0, inconsistencies=None):
    """Decorator to attach original class's docstring to the wrapped method.

    The output structure will be: top line of docstring, disclaimer about this
    being auto-derived, any extra text associated with the method being patched,
    the body of the docstring and finally, the list of keywords that exist in
    the original method but not in the dask version.

    Parameters
    ----------
    original_klass: type
        Original class which the method is derived from
    version : str
        Original package version which supports the wrapped method
    ua_args : list
        List of keywords which Dask doesn't support. Keywords existing in
        original but not in Dask will automatically be added.
    skipblocks : int
        How many text blocks (paragraphs) to skip from the start of the
        docstring. Useful for cases where the target has extra front-matter.
    inconsistencies: list
        List of known inconsistencies with method whose docstrings are being
        copied.
    """
    ua_args = ua_args or []

    def wrapper(method):
        try:
            extra = getattr(method, '__doc__', None) or ''
            method.__doc__ = _derived_from(original_klass, method, ua_args=ua_args, extra=extra, skipblocks=skipblocks, inconsistencies=inconsistencies)
            return method
        except AttributeError:
            module_name = original_klass.__module__.split('.')[0]

            @functools.wraps(method)
            def wrapped(*args, **kwargs):
                msg = f"Base package doesn't support '{method.__name__}'."
                if version is not None:
                    msg2 = ' Use {0} {1} or later to use this method.'
                    msg += msg2.format(module_name, version)
                raise NotImplementedError(msg)
            return wrapped
    return wrapper