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
def inner_test_loop_fn(A, dt, **kwargs):
    b_sizes = (1, 13)
    for b_size in b_sizes:
        b_order = next(cycle_order)
        B = self.specific_sample_matrix((A.shape[0], b_size), dt, b_order)
        check(A, B, **kwargs)
        b_order = next(cycle_order)
        tmp = B[:, 0].copy(order=b_order)
        check(A, tmp, **kwargs)