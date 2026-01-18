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
def check_1d(self, pyfunc):
    cfunc = nrtjit(pyfunc)
    n = 3
    expected = pyfunc(n)
    ret = cfunc(n)
    self.assert_array_nrt_refct(ret, 1)
    self.assertEqual(ret.size, expected.size)
    self.assertEqual(ret.shape, expected.shape)
    self.assertEqual(ret.dtype, expected.dtype)
    self.assertEqual(ret.strides, expected.strides)
    self.check_result_value(ret, expected)
    expected = np.empty_like(ret)
    expected.fill(123)
    ret.fill(123)
    np.testing.assert_equal(ret, expected)
    with self.assertRaises(ValueError) as cm:
        cfunc(-1)
    self.assertEqual(str(cm.exception), 'negative dimensions not allowed')