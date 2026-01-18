import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def test_digitize(self):
    pyfunc = digitize
    cfunc = jit(nopython=True)(pyfunc)

    def check(*args):
        expected = pyfunc(*args)
        got = cfunc(*args)
        self.assertPreciseEqual(expected, got)
    values = np.float64((0, 0.99, 1, 4.4, 4.5, 7, 8, 9, 9.5, float('inf'), float('-inf'), float('nan')))
    assert len(values) == 12
    self.rnd.shuffle(values)
    bins1 = np.float64([1, 3, 4.5, 8])
    bins2 = np.float64([1, 3, 4.5, 8, float('inf'), float('-inf')])
    bins3 = np.float64([1, 3, 4.5, 8, float('inf'), float('-inf')] + [float('nan')] * 10)
    all_bins = [bins1, bins2, bins3]
    xs = [values, values.reshape((3, 4))]
    for bins in all_bins:
        bins.sort()
        for x in xs:
            check(x, bins)
            check(x, bins[::-1])
    for bins in all_bins:
        for right in (True, False):
            check(values, bins, right)
            check(values, bins[::-1], right)
    check(list(values), bins1)
    check(np.array([np.nan, 1]), np.array([1.5, np.nan]))