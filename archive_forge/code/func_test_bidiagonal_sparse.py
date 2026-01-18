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
def test_bidiagonal_sparse(self):
    A = csc_matrix([[1, 3, 0], [0, 1, 5], [0, 0, 2]], dtype=float)
    e1 = math.exp(1)
    e2 = math.exp(2)
    expected = np.array([[e1, 3 * e1, 15 * (e2 - 2 * e1)], [0, e1, 5 * (e2 - e1)], [0, 0, e2]], dtype=float)
    observed = expm(A).toarray()
    assert_array_almost_equal(observed, expected)