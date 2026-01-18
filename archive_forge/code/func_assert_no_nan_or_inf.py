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
def assert_no_nan_or_inf(self, cfunc, args):
    msg = 'Array must not contain infs or NaNs.'
    self.assert_error(cfunc, args, msg, np.linalg.LinAlgError)