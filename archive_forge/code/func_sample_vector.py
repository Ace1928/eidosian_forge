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
def sample_vector(self, n, dtype):
    base = np.arange(n)
    if issubclass(dtype, np.complexfloating):
        return (base * (1 - 0.5j) + 2j).astype(dtype)
    else:
        return (base * 0.5 + 1).astype(dtype)