from numpy import array, arange, eye, zeros, ones, transpose, hstack
from numpy.linalg import norm
from numpy.testing import assert_allclose
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.linalg._interface import aslinearoperator
from scipy.sparse.linalg import lsmr
from .test_lsqr import G, b
def testColumnB(self):
    A = eye(self.n)
    b = ones((self.n, 1))
    x = lsmr(A, b)[0]
    assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)