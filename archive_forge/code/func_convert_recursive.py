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
def convert_recursive(is_match: Callable[[Any], bool], convert_item: Callable[[Any], Any], obj: Any) -> Any:
    """Either convert a single value if it matches a given function, or
    recursively walk over potentially nested lists, tuples and dicts applying
    the conversion, and returns the same type. Also supports the ArgsKwargs
    dataclass.
    """
    if is_match(obj):
        return convert_item(obj)
    elif isinstance(obj, ArgsKwargs):
        converted = convert_recursive(is_match, convert_item, list(obj.items()))
        return ArgsKwargs.from_items(converted)
    elif isinstance(obj, dict):
        converted = {}
        for key, value in obj.items():
            key = convert_recursive(is_match, convert_item, key)
            value = convert_recursive(is_match, convert_item, value)
            converted[key] = value
        return converted
    elif isinstance(obj, list):
        return [convert_recursive(is_match, convert_item, item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple((convert_recursive(is_match, convert_item, item) for item in obj))
    else:
        return obj