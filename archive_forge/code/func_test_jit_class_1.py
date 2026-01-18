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
def test_jit_class_1(self):
    Float2AndArray = self._make_Float2AndArray()
    Vector2 = self._make_Vector2()

    @njit
    def bar(obj):
        return obj.x + obj.y

    @njit
    def foo(a):
        obj = Float2AndArray(1, 2, a)
        obj.add(123)
        vec = Vector2(3, 4)
        return (bar(obj), bar(vec), obj.arr)
    inp = np.ones(10, dtype=np.float32)
    a, b, c = foo(inp)
    self.assertEqual(a, 123 + 1 + 123 + 2)
    self.assertEqual(b, 3 + 4)
    self.assertPreciseEqual(c, inp)