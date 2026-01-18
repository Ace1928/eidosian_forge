import contextlib
import sys
import numpy as np
import random
import re
import threading
import gc
from numba.core.errors import TypingError
from numba import njit
from numba.core import types, utils, config
from numba.tests.support import MemoryLeakMixin, TestCase, tag, skip_if_32bit
import unittest
def test_array_pass_through_sliced(self):

    def pyfunc(y):
        return y[y.size // 2:]
    arr = np.ones(4, dtype=np.float32)
    initrefct = sys.getrefcount(arr)
    cfunc = nrtjit(pyfunc)
    got = cfunc(arr)
    self.assertEqual(initrefct + 1, sys.getrefcount(arr))
    expected = pyfunc(arr)
    self.assertEqual(initrefct + 2, sys.getrefcount(arr))
    np.testing.assert_equal(expected, arr[arr.size // 2])
    np.testing.assert_equal(expected, got)
    del expected
    self.assertEqual(initrefct + 1, sys.getrefcount(arr))
    del got
    self.assertEqual(initrefct, sys.getrefcount(arr))