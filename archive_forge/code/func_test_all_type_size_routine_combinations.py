import itertools
import warnings
import numpy as np
from numpy import (arange, array, dot, zeros, identity, conjugate, transpose,
from numpy.random import random
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
from scipy.linalg import (solve, inv, det, lstsq, pinv, pinvh, norm,
from scipy.linalg._testutils import assert_no_overwrite
from scipy._lib._testutils import check_free_memory, IS_MUSL
from scipy.linalg.blas import HAS_ILP64
from scipy._lib.deprecation import _NoValue
@pytest.mark.skip(reason='Failure on OS X (gh-7500), crash on Windows (gh-8064)')
def test_all_type_size_routine_combinations(self):
    sizes = [10, 100]
    assume_as = ['gen', 'sym', 'pos', 'her']
    dtypes = [np.float32, np.float64, np.complex64, np.complex128]
    for size, assume_a, dtype in itertools.product(sizes, assume_as, dtypes):
        is_complex = dtype in (np.complex64, np.complex128)
        if assume_a == 'her' and (not is_complex):
            continue
        err_msg = f'Failed for size: {size}, assume_a: {assume_a},dtype: {dtype}'
        a = np.random.randn(size, size).astype(dtype)
        b = np.random.randn(size).astype(dtype)
        if is_complex:
            a = a + (1j * np.random.randn(size, size)).astype(dtype)
        if assume_a == 'sym':
            a = a + a.T
        elif assume_a == 'her':
            a = a + a.T.conj()
        elif assume_a == 'pos':
            a = a.conj().T.dot(a) + 0.1 * np.eye(size)
        tol = 1e-12 if dtype in (np.float64, np.complex128) else 1e-06
        if assume_a in ['gen', 'sym', 'her']:
            if dtype in (np.float32, np.complex64):
                tol *= 10
        x = solve(a, b, assume_a=assume_a)
        assert_allclose(a.dot(x), b, atol=tol * size, rtol=tol * size, err_msg=err_msg)
        if assume_a == 'sym' and dtype not in (np.complex64, np.complex128):
            x = solve(a, b, assume_a=assume_a, transposed=True)
            assert_allclose(a.dot(x), b, atol=tol * size, rtol=tol * size, err_msg=err_msg)