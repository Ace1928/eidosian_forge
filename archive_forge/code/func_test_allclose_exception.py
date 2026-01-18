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
def test_allclose_exception(self):
    self.disable_leak_check()
    pyfunc = np_allclose
    cfunc = jit(nopython=True)(pyfunc)
    inps = [(np.asarray([10000000000.0, 1e-09, np.nan]), np.asarray([10001000000.0, 1e-09]), 1e-05, 1e-08, False, 'shape mismatch: objects cannot be broadcast to a single shape', ValueError), ('hello', 3, False, 1e-08, False, 'The first argument "a" must be array-like', TypingError), (3, 'hello', False, 1e-08, False, 'The second argument "b" must be array-like', TypingError), (2, 3, False, 1e-08, False, 'The third argument "rtol" must be a floating point', TypingError), (2, 3, 1e-05, False, False, 'The fourth argument "atol" must be a floating point', TypingError), (2, 3, 1e-05, 1e-08, 1, 'The fifth argument "equal_nan" must be a boolean', TypingError)]
    for a, b, rtol, atol, equal_nan, exc_msg, exc in inps:
        with self.assertRaisesRegex(exc, exc_msg):
            cfunc(a, b, rtol, atol, equal_nan)