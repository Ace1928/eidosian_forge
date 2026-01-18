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
def test_diff1(self):
    pyfunc = diff1
    cfunc = jit(nopython=True)(pyfunc)
    for arr in self.diff_arrays():
        expected = pyfunc(arr)
        got = cfunc(arr)
        self.assertPreciseEqual(expected, got)
    a = np.array(42)
    with self.assertTypingError():
        cfunc(a)