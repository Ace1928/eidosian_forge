from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def base_test_arrays(dtype):
    if dtype == np.bool_:

        def factory(n):
            assert n % 2 == 0
            return np.bool_([0, 1] * (n // 2))
    else:

        def factory(n):
            return np.arange(n, dtype=dtype) + 1
    a1 = factory(10)
    a2 = factory(10).reshape(2, 5)
    a3 = factory(12)[::-1].reshape((2, 3, 2), order='A')
    assert not (a3.flags.c_contiguous or a3.flags.f_contiguous)
    return [a1, a2, a3]