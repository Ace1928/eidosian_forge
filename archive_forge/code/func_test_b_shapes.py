import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse.linalg import lsqr
def test_b_shapes():
    A = np.array([[1.0, 2.0]])
    b = 3.0
    x = lsqr(A, b)[0]
    assert norm(A.dot(x) - b) == pytest.approx(0)
    A = np.eye(10)
    b = np.ones((10, 1))
    x = lsqr(A, b)[0]
    assert norm(A.dot(x) - b.ravel()) == pytest.approx(0)