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
def test_isscalar(self):

    def values():
        yield 3
        yield np.asarray([3])
        yield (3,)
        yield 3j
        yield 'numba'
        yield int(10)
        yield np.int16(12345)
        yield 4.234
        yield True
        yield None
        yield np.timedelta64(10, 'Y')
        yield np.datetime64('nat')
        yield np.datetime64(1, 'Y')
    pyfunc = isscalar
    cfunc = jit(nopython=True)(pyfunc)
    for x in values():
        expected = pyfunc(x)
        got = cfunc(x)
        self.assertEqual(expected, got, x)