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
def benchmark(func, maxsec=1):
    timer = timeit.Timer(func)
    number = 1
    result = timer.repeat(1, number)
    while min(result) / number == 0:
        number *= 10
        result = timer.repeat(3, number)
    best = min(result) / number
    if best >= maxsec:
        return BenchmarkResult(func, result, number)
    max_per_run_time = maxsec / 3 / number
    number = max(max_per_run_time / best / 3, 1)
    number = int(10 ** math.ceil(math.log10(number)))
    records = timer.repeat(3, number)
    return BenchmarkResult(func, records, number)