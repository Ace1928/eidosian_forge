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
def partial_by_order(*args, **kwargs):
    """

    >>> from operator import add
    >>> partial_by_order(5, function=add, other=[(1, 10)])
    15
    """
    function = kwargs.pop('function')
    other = kwargs.pop('other')
    args2 = list(args)
    for i, arg in other:
        args2.insert(i, arg)
    return function(*args2, **kwargs)