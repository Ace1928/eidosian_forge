import numpy as np
from contextlib import contextmanager
import numba
from numba import njit, stencil
from numba.core import types, registry
from numba.core.compiler import compile_extra, Flags
from numba.core.cpu import ParallelOptions
from numba.tests.support import skip_parfors_unsupported, _32bit
from numba.core.errors import LoweringError, TypingError, NumbaValueError
import unittest
def test_impl4(n):
    A = np.arange(n)
    B = np.zeros(n)
    d = 1
    c = 2
    numba.stencil(lambda a, c, d: 0.3 * (a[-c + d] + a[0] + a[c - d]))(A, c, d, out=B)
    return B