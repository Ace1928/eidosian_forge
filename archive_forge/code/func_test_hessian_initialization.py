import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (BFGS, SR1)
def test_hessian_initialization(self):
    quasi_newton = (BFGS(), SR1())
    for qn in quasi_newton:
        qn.initialize(5, 'hess')
        B = qn.get_matrix()
        assert_array_equal(B, np.eye(5))