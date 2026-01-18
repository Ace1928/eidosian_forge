import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_check_derivative(self):

    def jac(x):
        return csr_matrix(self.jac(x))
    accuracy = check_derivative(self.fun, jac, self.x0, bounds=(self.lb, self.ub))
    assert_(accuracy < 1e-09)
    accuracy = check_derivative(self.fun, jac, self.x0, bounds=(self.lb, self.ub))
    assert_(accuracy < 1e-09)