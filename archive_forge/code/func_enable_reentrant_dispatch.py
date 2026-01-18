import __future__  # noqa: F404
import collections
import functools
import types
import warnings
from typing import Dict, Set, List, Any, Callable, Iterable, Type, Tuple
from functools import wraps
import contextlib
import torch
from torch._C import (
@contextlib.contextmanager
def enable_reentrant_dispatch():
    with torch._C._RestorePythonTLSSnapshot():
        try:
            yield
        finally:
            pass