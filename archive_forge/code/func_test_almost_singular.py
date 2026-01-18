import pytest
import numpy as np
from numpy.linalg import lstsq
from numpy.testing import assert_allclose, assert_equal, assert_
from scipy.sparse import rand, coo_matrix
from scipy.sparse.linalg import aslinearoperator
from scipy.optimize import lsq_linear
from scipy.optimize._minimize import Bounds
def test_almost_singular(self):
    A = np.array([[0.8854232310355122, 0.0365312146937765, 0.0365312146836789], [0.3742460132129041, 0.0130523214078376, 0.0130523214077873], [0.9680633871281361, 0.0319366128718639, 0.0319366128718388]])
    b = np.array([0.0055029366538097, 0.0026677442422208, 0.0066612514782381])
    result = lsq_linear(A, b, method=self.method)
    assert_(result.cost < 1.1e-08)