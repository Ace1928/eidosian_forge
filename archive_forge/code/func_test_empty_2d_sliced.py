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
def test_empty_2d_sliced(self):

    def pyfunc(m, n, p):
        arr = np.empty((m, n), np.int32)
        for i in range(m):
            for j in range(n):
                arr[i, j] = i + j
        return arr[p]
    cfunc = nrtjit(pyfunc)
    m = 4
    n = 3
    p = 2
    expected_arr = pyfunc(m, n, p)
    got_arr = cfunc(m, n, p)
    self.assert_array_nrt_refct(got_arr, 1)
    np.testing.assert_equal(expected_arr, got_arr)
    self.assertEqual(expected_arr.size, got_arr.size)
    self.assertEqual(expected_arr.shape, got_arr.shape)
    self.assertEqual(expected_arr.strides, got_arr.strides)
    del got_arr