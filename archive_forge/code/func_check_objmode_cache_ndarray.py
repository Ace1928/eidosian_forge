import inspect
import math
import operator
import sys
import pickle
import multiprocessing
import ctypes
import warnings
import re
import numpy as np
from llvmlite import ir
import numba
from numba import njit, jit, vectorize, guvectorize, objmode
from numba.core import types, errors, typing, compiler, cgutils
from numba.core.typed_passes import type_inference_stage
from numba.core.registry import cpu_target
from numba.core.imputils import lower_constant
from numba.tests.support import (
from numba.core.errors import LoweringError
import unittest
from numba.extending import (
from numba.core.typing.templates import (
from .pdlike_usecase import Index, Series
@classmethod
def check_objmode_cache_ndarray(cls):

    def do_this(a, b):
        return np.sum(a + b)

    def do_something(a, b):
        return np.sum(a + b)

    @overload(do_something)
    def overload_do_something(a, b):

        def _do_something_impl(a, b):
            with objmode(y='float64'):
                y = do_this(a, b)
            return y
        return _do_something_impl

    @njit(cache=True)
    def test_caching():
        a = np.arange(20)
        b = np.arange(20)
        return do_something(a, b)
    got = test_caching()
    expect = test_caching.py_func()
    if got != expect:
        raise AssertionError('incorrect result')
    return test_caching