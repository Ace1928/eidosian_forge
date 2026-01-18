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
def assert_mismatching_dtypes(self, cfunc, args, func_name='np.dot()'):
    with self.assertRaises(errors.TypingError) as raises:
        cfunc(*args)
    self.assertIn('%s arguments must all have the same dtype' % (func_name,), str(raises.exception))