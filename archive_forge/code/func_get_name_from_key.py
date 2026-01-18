from __future__ import annotations
import dataclasses
import datetime
import decimal
import hashlib
import inspect
import pathlib
import pickle
import types
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from functools import partial
from numbers import Integral, Number
from operator import getitem
from typing import Any, Literal, TypeVar
import cloudpickle
from tlz import curry, groupby, identity, merge
from tlz.functoolz import Compose
from dask import config, local
from dask._compatibility import EMSCRIPTEN
from dask.core import flatten
from dask.core import get as simple_get
from dask.core import literal, quote
from dask.hashing import hash_buffer_hex
from dask.system import CPU_COUNT
from dask.typing import Key, SchedulerGetCallable
from dask.utils import (
def get_name_from_key(key: Key) -> str:
    """Given a dask collection's key, extract the collection name.

    Parameters
    ----------
    key: string or tuple
        Dask collection's key, which must be either a single string or a tuple whose
        first element is a string (commonly referred to as a collection's 'name'),

    Examples
    --------
    >>> get_name_from_key("foo")
    'foo'
    >>> get_name_from_key(("foo-123", 1, 2))
    'foo-123'
    """
    if isinstance(key, tuple) and key and isinstance(key[0], str):
        return key[0]
    if isinstance(key, str):
        return key
    raise TypeError(f'Expected str or a tuple starting with str; got {key!r}')