import ctypes
import itertools
import pickle
import random
import typing as pt
import unittest
from collections import OrderedDict
import numpy as np
from numba import (boolean, deferred_type, float32, float64, int16, int32,
from numba.core import errors, types
from numba.core.dispatcher import Dispatcher
from numba.core.errors import LoweringError, TypingError
from numba.core.runtime.nrt import MemInfo
from numba.experimental import jitclass
from numba.experimental.jitclass import _box
from numba.experimental.jitclass.base import JitClassType
from numba.tests.support import MemoryLeakMixin, TestCase, skip_if_typeguard
from numba.tests.support import skip_unless_scipy
def test_arithmetic_logical(self):
    IntWrapper = self.get_int_wrapper()
    FloatWrapper = self.get_float_wrapper()
    float_py_funcs = [lambda x, y: x == y, lambda x, y: x != y, lambda x, y: x >= y, lambda x, y: x > y, lambda x, y: x <= y, lambda x, y: x < y, lambda x, y: x + y, lambda x, y: x // y, lambda x, y: x % y, lambda x, y: x * y, lambda x, y: x ** y, lambda x, y: x - y, lambda x, y: x / y]
    int_py_funcs = [lambda x, y: x == y, lambda x, y: x != y, lambda x, y: x << y, lambda x, y: x >> y, lambda x, y: x & y, lambda x, y: x | y, lambda x, y: x ^ y]
    test_values = [(0.0, 2.0), (1.234, 3.1415), (13.1, 1.01)]

    def unwrap(value):
        return getattr(value, 'x', value)
    for jit_f, (x, y) in itertools.product(map(njit, float_py_funcs), test_values):
        py_f = jit_f.py_func
        expected = py_f(x, y)
        jit_x = FloatWrapper(x)
        jit_y = FloatWrapper(y)
        check = self.assertEqual if type(expected) is not float else self.assertAlmostEqual
        check(expected, jit_f(x, y))
        check(expected, unwrap(py_f(jit_x, jit_y)))
        check(expected, unwrap(jit_f(jit_x, jit_y)))
    for jit_f, (x, y) in itertools.product(map(njit, int_py_funcs), test_values):
        py_f = jit_f.py_func
        x, y = (int(x), int(y))
        expected = py_f(x, y)
        jit_x = IntWrapper(x)
        jit_y = IntWrapper(y)
        self.assertEqual(expected, jit_f(x, y))
        self.assertEqual(expected, unwrap(py_f(jit_x, jit_y)))
        self.assertEqual(expected, unwrap(jit_f(jit_x, jit_y)))