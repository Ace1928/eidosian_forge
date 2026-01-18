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
def test_burkardt_12(self):
    A = np.array([[-131, 19, 18], [-390, 56, 54], [-387, 57, 52]], dtype=float)
    assert_allclose(sorted(scipy.linalg.eigvals(A)), (-20, -2, -1))
    desired = np.array([[-1.509644158793135, 0.3678794391096522, 0.1353352811751005], [-5.632570799891469, 1.471517758499875, 0.4060058435250609], [-4.934938326088363, 1.103638317328798, 0.5413411267617766]], dtype=float)
    actual = expm(A)
    assert_allclose(actual, desired)