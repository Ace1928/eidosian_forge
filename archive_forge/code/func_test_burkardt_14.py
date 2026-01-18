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
def test_burkardt_14(self):
    A = np.array([[0, 1e-08, 0], [-(20000000000.0 + 400000000.0 / 6.0), -3, 20000000000.0], [200.0 / 3.0, 0, -200.0 / 3.0]], dtype=float)
    desired = np.array([[0.446849468283175, 1.54044157383952e-09, 0.462811453558774], [-5743067.77947947, -0.0152830038686819, -4526542.71278401], [0.447722977849494, 1.54270484519591e-09, 0.463480648837651]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)