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
def test_digitize_supplemental(self):
    pyfunc = digitize
    cfunc = jit(nopython=True)(pyfunc)

    def check(*args):
        expected = pyfunc(*args)
        got = cfunc(*args)
        self.assertPreciseEqual(expected, got)
    x = np.arange(-6, 5)
    bins = np.arange(-5, 5)
    check(x, bins)
    x = np.arange(5, -6, -1)
    bins = np.arange(5, -5, -1)
    check(x, bins)
    x = self.rnd.rand(10)
    bins = np.linspace(x.min(), x.max(), 10)
    check(x, bins)
    x = [1, 5, 4, 10, 8, 11, 0]
    bins = [1, 5, 10]
    check(x, bins)
    x = np.arange(-6, 5)
    bins = np.arange(-6, 4)
    check(x, bins, True)
    x = np.arange(5, -6, -1)
    bins = np.arange(4, -6, -1)
    check(x, bins, True)
    x = self.rnd.rand(10)
    bins = np.linspace(x.min(), x.max(), 10)
    check(x, bins, True)
    x = [-1, 0, 1, 2]
    bins = [0, 0, 1]
    check(x, bins)
    bins = [1, 1, 0]
    check(x, bins)
    bins = [1, 1, 1, 1]
    check(x, bins)
    x = 2 ** 54
    check([x], [x - 1, x + 1])