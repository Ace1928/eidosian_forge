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
def test_bool_fallback(self):

    def py_b(x):
        return bool(x)
    jit_b = njit(py_b)

    @jitclass([('x', types.List(types.intp))])
    class LenClass:

        def __init__(self, x):
            self.x = x

        def __len__(self):
            return len(self.x) % 4

        def append(self, y):
            self.x.append(y)

        def pop(self):
            self.x.pop(0)
    obj = LenClass([1, 2, 3])
    self.assertTrue(py_b(obj))
    self.assertTrue(jit_b(obj))
    obj.append(4)
    self.assertFalse(py_b(obj))
    self.assertFalse(jit_b(obj))
    obj.pop()
    self.assertTrue(py_b(obj))
    self.assertTrue(jit_b(obj))

    @jitclass([('y', types.float64)])
    class NormalClass:

        def __init__(self, y):
            self.y = y
    obj = NormalClass(0)
    self.assertTrue(py_b(obj))
    self.assertTrue(jit_b(obj))