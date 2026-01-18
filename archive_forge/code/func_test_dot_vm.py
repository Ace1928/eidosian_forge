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
@needs_blas
def test_dot_vm(self):
    """
        Test vector * matrix and matrix * vector np.dot()
        """
    self.check_dot_vm(dot2, dot3, 'np.dot()')