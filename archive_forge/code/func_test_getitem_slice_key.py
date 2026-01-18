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
def test_getitem_slice_key(self):
    spec = [('data', int32[:])]

    @jitclass(spec)
    class TestClass(object):

        def __init__(self):
            self.data = np.zeros(10, dtype=np.int32)

        def __setitem__(self, slc, data):
            self.data[slc.start] = data
            self.data[slc.stop] = data + slc.step

        def __getitem__(self, slc):
            return self.data[slc.start]
    t = TestClass()
    t[1:5:1] = 1
    self.assertEqual(t[1:1:1], 1)
    self.assertEqual(t[5:5:5], 2)

    @njit
    def get5(t):
        return t[5:6:1]
    self.assertEqual(get5(t), 2)

    @njit
    def set26(t, data):
        t[2:6:1] = data
    set26(t, 2)
    self.assertEqual(t[2:2:1], 2)
    self.assertEqual(t[6:6:1], 3)