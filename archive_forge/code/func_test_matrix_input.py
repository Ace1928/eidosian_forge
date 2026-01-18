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
def test_matrix_input(self):
    A = np.zeros((200, 200))
    A[-1, 0] = 1
    B0 = expm(A)
    with suppress_warnings() as sup:
        sup.filter(DeprecationWarning, 'the matrix subclass.*')
        sup.filter(PendingDeprecationWarning, 'the matrix subclass.*')
        B = expm(np.matrix(A))
    assert_allclose(B, B0)