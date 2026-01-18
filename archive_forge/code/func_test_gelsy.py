import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
def test_gelsy(self):
    for dtype in REAL_DTYPES:
        a1 = np.array([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]], dtype=dtype)
        b1 = np.array([16.0, 17.0, 20.0], dtype=dtype)
        gelsy, gelsy_lwork = get_lapack_funcs(('gelsy', 'gelss_lwork'), (a1, b1))
        m, n = a1.shape
        if len(b1.shape) == 2:
            nrhs = b1.shape[1]
        else:
            nrhs = 1
        work, info = gelsy_lwork(m, n, nrhs, 10 * np.finfo(dtype).eps)
        lwork = int(np.real(work))
        jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
        v, x, j, rank, info = gelsy(a1, b1, jptv, np.finfo(dtype).eps, lwork, False, False)
        assert_allclose(x[:-1], np.array([-14.333333333333323, 14.999999999999991], dtype=dtype), rtol=25 * np.finfo(dtype).eps)
    for dtype in COMPLEX_DTYPES:
        a1 = np.array([[1.0 + 4j, 2.0], [4.0 + 0.5j, 5.0 - 3j], [7.0 - 2j, 8.0 + 0.7j]], dtype=dtype)
        b1 = np.array([16.0, 17.0 + 2j, 20.0 - 4j], dtype=dtype)
        gelsy, gelsy_lwork = get_lapack_funcs(('gelsy', 'gelss_lwork'), (a1, b1))
        m, n = a1.shape
        if len(b1.shape) == 2:
            nrhs = b1.shape[1]
        else:
            nrhs = 1
        work, info = gelsy_lwork(m, n, nrhs, 10 * np.finfo(dtype).eps)
        lwork = int(np.real(work))
        jptv = np.zeros((a1.shape[1], 1), dtype=np.int32)
        v, x, j, rank, info = gelsy(a1, b1, jptv, np.finfo(dtype).eps, lwork, False, False)
        assert_allclose(x[:-1], np.array([1.161753632288328 - 1.901075709391912j, 1.735882340522193 + 1.521240901196909j], dtype=dtype), rtol=25 * np.finfo(dtype).eps)