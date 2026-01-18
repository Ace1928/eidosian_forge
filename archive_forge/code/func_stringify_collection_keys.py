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
def stringify_collection_keys(obj):
    """Convert all collection keys in ``obj`` to strings.

    This is a specialized version of ``stringify()`` that only converts keys
    of the form: ``("a string", ...)``
    """
    typ = type(obj)
    if typ is tuple and obj:
        obj0 = obj[0]
        if type(obj0) is str or type(obj0) is bytes:
            return stringify(obj)
        if callable(obj0):
            return (obj0,) + tuple((stringify_collection_keys(x) for x in obj[1:]))
    if typ is list:
        return [stringify_collection_keys(v) for v in obj]
    if typ is dict:
        return {k: stringify_collection_keys(v) for k, v in obj.items()}
    if typ is tuple:
        return tuple((stringify_collection_keys(v) for v in obj))
    return obj