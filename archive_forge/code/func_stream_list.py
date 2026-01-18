import atexit
import builtins
import functools
import inspect
import os
import operator
import timeit
import math
import sys
import traceback
import weakref
import warnings
import threading
import contextlib
import typing as _tp
from types import ModuleType
from importlib import import_module
import numpy as np
from inspect import signature as pysignature # noqa: F401
from inspect import Signature as pySignature # noqa: F401
from inspect import Parameter as pyParameter # noqa: F401
from numba.core.config import (PYVERSION, MACHINE_BITS, # noqa: F401
from numba.core import config
from numba.core import types
from collections.abc import Mapping, Sequence, MutableSet, MutableMapping
def stream_list(lst):
    """
    Given a list, return an infinite iterator of iterators.
    Each iterator iterates over the list from the last seen point up to
    the current end-of-list.

    In effect, each iterator will give the newly appended elements from the
    previous iterator instantiation time.
    """

    def sublist_iterator(start, stop):
        return iter(lst[start:stop])
    start = 0
    while True:
        stop = len(lst)
        yield sublist_iterator(start, stop)
        start = stop