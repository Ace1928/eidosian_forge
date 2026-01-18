import sys
import subprocess
from itertools import product
from textwrap import dedent
import numpy as np
from numba import config
from numba import njit
from numba import int32, float32, prange, uint8
from numba.core import types
from numba import typeof
from numba.typed import List, Dict
from numba.core.errors import TypingError
from numba.tests.support import (TestCase, MemoryLeakMixin, override_config,
from numba.core.unsafe.refcount import get_refcount
from numba.experimental import jitclass
def test_jitclass_as_item_in_list(self):
    spec = [('value', int32), ('array', float32[:])]

    @jitclass(spec)
    class Bag(object):

        def __init__(self, value):
            self.value = value
            self.array = np.zeros(value, dtype=np.float32)

        @property
        def size(self):
            return self.array.size

        def increment(self, val):
            for i in range(self.size):
                self.array[i] += val
            return self.array

    @njit
    def foo():
        l = List()
        l.append(Bag(21))
        l.append(Bag(22))
        l.append(Bag(23))
        return l
    expected = foo.py_func()
    got = foo()

    def bag_equal(one, two):
        self.assertEqual(one.value, two.value)
        np.testing.assert_allclose(one.array, two.array)
    [bag_equal(a, b) for a, b in zip(expected, got)]