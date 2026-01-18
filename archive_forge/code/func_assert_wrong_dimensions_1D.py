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
def assert_wrong_dimensions_1D(self, name, cfunc, args, la_prefix=True):
    prefix = 'np.linalg' if la_prefix else 'np'
    msg = '%s.%s() only supported on 1 and 2-D arrays' % (prefix, name)
    self.assert_error(cfunc, args, msg, errors.TypingError)