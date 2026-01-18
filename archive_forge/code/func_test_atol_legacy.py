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
def test_atol_legacy(self):
    A = eye(2)
    b = ones(2)
    x, info = gmres(A, b, rtol=1e-05)
    assert np.linalg.norm(A @ x - b) <= 1e-05 * np.linalg.norm(b)
    assert_allclose(x, b, atol=0, rtol=1e-08)
    rndm = np.random.RandomState(12345)
    A = rndm.rand(30, 30)
    b = 1e-06 * ones(30)
    x, info = gmres(A, b, rtol=1e-07, restart=20)
    assert np.linalg.norm(A @ x - b) > 1e-07
    A = eye(2)
    b = 1e-10 * ones(2)
    x, info = gmres(A, b, rtol=1e-08, atol=0)
    assert np.linalg.norm(A @ x - b) <= 1e-08 * np.linalg.norm(b)