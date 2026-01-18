import contextlib
import functools
import operator
import platform
import itertools
import sys
from scipy._lib import _pep440
import numpy as np
from numpy import (arange, zeros, array, dot, asarray,
import random
from numpy.testing import (assert_equal, assert_array_equal,
from pytest import raises as assert_raises
import scipy.linalg
import scipy.sparse as sparse
from scipy.sparse import (csc_matrix, csr_matrix, dok_matrix,
from scipy.sparse._sputils import (supported_dtypes, isscalarlike,
from scipy.sparse.linalg import splu, expm, inv
from scipy._lib.decorator import decorator
from scipy._lib._util import ComplexWarning
import pytest
def test_matmat_sparse(self):
    a = matrix([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]])
    a2 = array([[3, 0, 0], [0, 1, 0], [2, 0, 3.0], [2, 3, 0]])
    b = matrix([[0, 1], [1, 0], [0, 2]], 'd')
    asp = self.spcreator(a)
    bsp = self.spcreator(b)
    assert_array_almost_equal((asp @ bsp).toarray(), a @ b)
    assert_array_almost_equal(asp @ b, a @ b)
    assert_array_almost_equal(a @ bsp, a @ b)
    assert_array_almost_equal(a2 @ bsp, a @ b)
    csp = bsp.tocsc()
    c = b
    want = a @ c
    assert_array_almost_equal((asp @ csp).toarray(), want)
    assert_array_almost_equal(asp @ c, want)
    assert_array_almost_equal(a @ csp, want)
    assert_array_almost_equal(a2 @ csp, want)
    csp = bsp.tocsr()
    assert_array_almost_equal((asp @ csp).toarray(), want)
    assert_array_almost_equal(asp @ c, want)
    assert_array_almost_equal(a @ csp, want)
    assert_array_almost_equal(a2 @ csp, want)
    csp = bsp.tocoo()
    assert_array_almost_equal((asp @ csp).toarray(), want)
    assert_array_almost_equal(asp @ c, want)
    assert_array_almost_equal(a @ csp, want)
    assert_array_almost_equal(a2 @ csp, want)
    L = 30
    frac = 0.3
    random.seed(0)
    A = zeros((L, 2))
    for i in range(L):
        for j in range(2):
            r = random.random()
            if r < frac:
                A[i, j] = r / frac
    A = self.spcreator(A)
    B = A @ A.T
    assert_array_almost_equal(B.toarray(), A.toarray() @ A.T.toarray())
    assert_array_almost_equal(B.toarray(), A.toarray() @ A.toarray().T)
    A = self.spcreator([[1, 2], [3, 4]])
    B = self.spcreator([[1, 2], [3, 4], [5, 6]])
    assert_raises(ValueError, A.__matmul__, B)
    if isinstance(A, sparray):
        assert_raises(ValueError, A.__mul__, B)