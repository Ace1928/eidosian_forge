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
def test_interp_float_precision_handled_per_numpy(self):
    pyfunc = interp
    cfunc = jit(nopython=True)(pyfunc)
    dtypes = [np.float32, np.float64, np.int32, np.int64]
    for combo in itertools.combinations_with_replacement(dtypes, 3):
        xp_dtype, fp_dtype, x_dtype = combo
        xp = np.arange(10, dtype=xp_dtype)
        fp = (xp ** 2).astype(fp_dtype)
        x = np.linspace(2, 3, 10, dtype=x_dtype)
        expected = pyfunc(x, xp, fp)
        got = cfunc(x, xp, fp)
        self.assertPreciseEqual(expected, got)