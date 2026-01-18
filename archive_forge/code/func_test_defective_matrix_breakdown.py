import itertools
import platform
import sys
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numpy import zeros, arange, array, ones, eye, iscomplexobj
from numpy.linalg import norm
from scipy.sparse import spdiags, csr_matrix, kronsum
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,
def test_defective_matrix_breakdown(self):
    A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    b = np.array([1, 0, 1])
    rtol = 1e-08
    x, info = gmres(A, b, rtol=rtol, atol=0)
    assert not np.isnan(x).any()
    if info == 0:
        assert np.linalg.norm(A @ x - b) <= rtol * np.linalg.norm(b)
    assert_allclose(A @ (A @ x), A @ b)