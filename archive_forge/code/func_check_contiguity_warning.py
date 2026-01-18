import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
@contextlib.contextmanager
def check_contiguity_warning(self, pyfunc):
    """
        Check performance warning(s) for non-contiguity.
        """
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always', errors.NumbaPerformanceWarning)
        yield
    self.assertGreaterEqual(len(w), 1)
    self.assertIs(w[0].category, errors.NumbaPerformanceWarning)
    self.assertIn('faster on contiguous arrays', str(w[0].message))
    self.assertEqual(w[0].filename, pyfunc.__code__.co_filename)
    self.assertEqual(w[0].lineno, pyfunc.__code__.co_firstlineno + 1)