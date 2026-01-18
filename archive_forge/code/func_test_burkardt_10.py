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
def test_burkardt_10(self):
    A = np.array([[4, 2, 0], [1, 4, 1], [1, 1, 4]], dtype=float)
    assert_allclose(sorted(scipy.linalg.eigvals(A)), (3, 3, 6))
    desired = np.array([[147.8666224463699, 183.7651386463682, 71.79703239999647], [127.7810855231823, 183.7651386463682, 91.88256932318416], [127.7810855231824, 163.6796017231806, 111.9681062463718]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)