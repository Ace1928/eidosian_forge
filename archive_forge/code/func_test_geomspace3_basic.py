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
def test_geomspace3_basic(self):

    def inputs():
        yield (-1, -60, 50)
        yield (1.0, 60.0, 70)
        yield (-60.0, -1.0, 80)
        yield (1, 1000, 4)
        yield (1, 1000, 3)
        yield (1000, 1, 4)
        yield (1, 256, 9)
        yield (-1000, -1, 4)
        yield (-1, np.complex64(2j), 10)
        yield (np.complex64(2j), -1, 20)
        yield (-1.0, np.complex64(2j), 30)
        yield (np.complex64(1j), np.complex64(1000j), 4)
        yield (np.complex64(-1 + 0j), np.complex64(1 + 0j), 5)
        yield (np.complex64(1), np.complex64(2), 40)
        yield (np.complex64(2j), np.complex64(4j), 50)
        yield (np.complex64(2), np.complex64(4j), 60)
        yield (np.complex64(1 + 2j), np.complex64(3 + 4j), 70)
        yield (np.complex64(1 - 2j), np.complex64(3 - 4j), 80)
        yield (np.complex64(-1 + 2j), np.complex64(3 + 4j), 90)
    pyfunc = geomspace3
    cfunc = jit(nopython=True)(pyfunc)
    for start, stop, num in inputs():
        self.assertPreciseEqual(pyfunc(start, stop, num), cfunc(start, stop, num), abs_tol=1e-14)