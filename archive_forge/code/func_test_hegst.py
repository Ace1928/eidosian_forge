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
def test_hegst():
    seed(1234)
    for ind, dtype in enumerate(COMPLEX_DTYPES):
        n = 10
        potrf, hegst, heevd, hegvd = get_lapack_funcs(('potrf', 'hegst', 'heevd', 'hegvd'), dtype=dtype)
        A = rand(n, n).astype(dtype) + 1j * rand(n, n).astype(dtype)
        A = (A + A.conj().T) / 2
        B = rand(n, n).astype(dtype) + 1j * rand(n, n).astype(dtype)
        B = (B + B.conj().T) / 2 + 2 * np.eye(n, dtype=dtype)
        eig_gvd, _, info = hegvd(A, B)
        assert_(info == 0)
        b, info = potrf(B)
        assert_(info == 0)
        a, info = hegst(A, b)
        assert_(info == 0)
        eig, _, info = heevd(a)
        assert_(info == 0)
        assert_allclose(eig, eig_gvd, rtol=0.0001)