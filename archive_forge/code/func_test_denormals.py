from numpy.testing import (assert_, assert_allclose, assert_equal,
import pytest
from platform import python_implementation
import numpy as np
from numpy import zeros, array, allclose
from scipy.linalg import norm
from scipy.sparse import csr_matrix, eye, rand
from scipy.sparse.linalg._interface import LinearOperator
from scipy.sparse.linalg import splu
from scipy.sparse.linalg._isolve import lgmres, gmres
def test_denormals(self):
    A = np.array([[1, 2], [3, 4]], dtype=float)
    A *= 100 * np.nextafter(0, 1)
    b = np.array([1, 1])
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, '.*called without specifying.*')
        xp, info = lgmres(A, b)
    if info == 0:
        assert_allclose(A.dot(xp), b)