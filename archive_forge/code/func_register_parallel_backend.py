from __future__ import division
import os
import sys
from math import sqrt
import functools
import collections
import time
import threading
import itertools
from uuid import uuid4
from numbers import Integral
import warnings
import queue
import weakref
from contextlib import nullcontext
from multiprocessing import TimeoutError
from ._multiprocessing_helpers import mp
from .logger import Logger, short_format_time
from .disk import memstr_to_bytes
from ._parallel_backends import (FallbackToBackend, MultiprocessingBackend,
from ._utils import eval_expr, _Sentinel
from ._parallel_backends import AutoBatchingMixin  # noqa
from ._parallel_backends import ParallelBackendBase  # noqa
def register_parallel_backend(name, factory, make_default=False):
    """Register a new Parallel backend factory.

    The new backend can then be selected by passing its name as the backend
    argument to the :class:`~Parallel` class. Moreover, the default backend can
    be overwritten globally by setting make_default=True.

    The factory can be any callable that takes no argument and return an
    instance of ``ParallelBackendBase``.

    Warning: this function is experimental and subject to change in a future
    version of joblib.

    .. versionadded:: 0.10
    """
    BACKENDS[name] = factory
    if make_default:
        global DEFAULT_BACKEND
        DEFAULT_BACKEND = name