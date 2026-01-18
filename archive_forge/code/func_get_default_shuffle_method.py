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
def get_default_shuffle_method() -> str:
    if (d := config.get('dataframe.shuffle.method', None)):
        return d
    try:
        from distributed import default_client
        default_client()
    except (ImportError, ValueError):
        return 'disk'
    try:
        from distributed.shuffle import check_minimal_arrow_version
        check_minimal_arrow_version()
    except ModuleNotFoundError:
        return 'tasks'
    return 'p2p'