import math
import numpy as np
from numpy import array, eye, exp, random
from numpy.testing import (
from scipy.sparse import csc_matrix, csc_array, SparseEfficiencyWarning
from scipy.sparse._construct import eye as speye
from scipy.sparse.linalg._matfuncs import (expm, _expm,
from scipy.sparse._sputils import matrix
from scipy.linalg import logm
from scipy.special import factorial, binom
import scipy.sparse
import scipy.sparse.linalg
def test_burkardt_4(self):
    A = np.array([[-49, 24], [-64, 31]], dtype=float)
    U = np.array([[3, 1], [4, 2]], dtype=float)
    V = np.array([[1, -1 / 2], [-2, 3 / 2]], dtype=float)
    w = np.array([-17, -1], dtype=float)
    desired = np.dot(U * np.exp(w), V)
    actual = expm(A)
    assert_allclose(actual, desired)