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
def test_logm_consistency(self):
    random.seed(1234)
    for dtype in [np.float64, np.complex128]:
        for n in range(1, 10):
            for scale in [0.0001, 0.001, 0.01, 0.1, 1, 10.0, 100.0]:
                A = (eye(n) + random.rand(n, n) * scale).astype(dtype)
                if np.iscomplexobj(A):
                    A = A + 1j * random.rand(n, n) * scale
                assert_array_almost_equal(expm(logm(A)), A)