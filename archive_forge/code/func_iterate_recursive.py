import contextlib
import functools
import inspect
import os
import platform
import random
import tempfile
import threading
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
import numpy
from packaging.version import Version
from wasabi import table
from .compat import (
from .compat import mxnet as mx
from .compat import tensorflow as tf
from .compat import torch
from typing import TYPE_CHECKING
from . import types  # noqa: E402
from .types import ArgsKwargs, ArrayXd, FloatsXd, IntsXd, Padded, Ragged  # noqa: E402
def iterate_recursive(is_match: Callable[[Any], bool], obj: Any) -> Any:
    """Either yield a single value if it matches a given function, or recursively
    walk over potentially nested lists, tuples and dicts yielding matching
    values. Also supports the ArgsKwargs dataclass.
    """
    if is_match(obj):
        yield obj
    elif isinstance(obj, ArgsKwargs):
        yield from iterate_recursive(is_match, list(obj.items()))
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from iterate_recursive(is_match, key)
            yield from iterate_recursive(is_match, value)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for item in obj:
            yield from iterate_recursive(is_match, item)