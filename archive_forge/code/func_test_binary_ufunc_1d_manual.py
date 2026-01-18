import itertools
import pytest
import numpy as np
from numpy.core._multiarray_tests import solve_diophantine, internal_overlap
from numpy.core import _umath_tests
from numpy.lib.stride_tricks import as_strided
from numpy.testing import (
@pytest.mark.slow
def test_binary_ufunc_1d_manual(self):
    ufunc = np.add

    def check(a, b, c):
        c0 = c.copy()
        c1 = ufunc(a, b, out=c0)
        c2 = ufunc(a, b, out=c)
        assert_array_equal(c1, c2)
    for dtype in [np.int8, np.int16, np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]:
        n = 1000
        k = 10
        indices = []
        for p in [1, 2]:
            indices.extend([np.index_exp[:p * n:p], np.index_exp[k:k + p * n:p], np.index_exp[p * n - 1::-p], np.index_exp[k + p * n - 1:k - 1:-p]])
        for x, y, z in itertools.product(indices, indices, indices):
            v = np.arange(6 * n).astype(dtype)
            x = v[x]
            y = v[y]
            z = v[z]
            check(x, y, z)
            check(x[:1], y, z)
            check(x[-1:], y, z)
            check(x[:1].reshape([]), y, z)
            check(x[-1:].reshape([]), y, z)
            check(x, y[:1], z)
            check(x, y[-1:], z)
            check(x, y[:1].reshape([]), z)
            check(x, y[-1:].reshape([]), z)