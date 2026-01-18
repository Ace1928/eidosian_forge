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
def test_arithmetic_logical_reflection(self):

    class OperatorsDefined:

        def __init__(self, x):
            self.x = x

        def __radd__(self, other):
            return other.x + self.x

        def __rsub__(self, other):
            return other.x - self.x

        def __rmul__(self, other):
            return other.x * self.x

        def __rtruediv__(self, other):
            return other.x / self.x

        def __rfloordiv__(self, other):
            return other.x // self.x

        def __rmod__(self, other):
            return other.x % self.x

        def __rpow__(self, other):
            return other.x ** self.x

        def __rlshift__(self, other):
            return other.x << self.x

        def __rrshift__(self, other):
            return other.x >> self.x

        def __rand__(self, other):
            return other.x & self.x

        def __rxor__(self, other):
            return other.x ^ self.x

        def __ror__(self, other):
            return other.x | self.x

    class NoOperatorsDefined:

        def __init__(self, x):
            self.x = x
    float_op = ['+', '-', '*', '**', '/', '//', '%']
    int_op = [*float_op, '<<', '>>', '&', '^', '|']
    for test_type, test_op, test_value in [(int32, int_op, (2, 4)), (float64, float_op, (2.0, 4.0)), (float64[::1], float_op, (np.array([1.0, 2.0, 4.0]), np.array([20.0, -24.0, 1.0])))]:
        spec = {'x': test_type}
        JitOperatorsDefined = jitclass(OperatorsDefined, spec)
        JitNoOperatorsDefined = jitclass(NoOperatorsDefined, spec)
        py_ops_defined = OperatorsDefined(test_value[0])
        py_ops_not_defined = NoOperatorsDefined(test_value[1])
        jit_ops_defined = JitOperatorsDefined(test_value[0])
        jit_ops_not_defined = JitNoOperatorsDefined(test_value[1])
        for op in test_op:
            if not 'array' in str(test_type):
                self.assertEqual(eval(f'py_ops_not_defined {op} py_ops_defined'), eval(f'jit_ops_not_defined {op} jit_ops_defined'))
            else:
                self.assertTupleEqual(tuple(eval(f'py_ops_not_defined {op} py_ops_defined')), tuple(eval(f'jit_ops_not_defined {op} jit_ops_defined')))