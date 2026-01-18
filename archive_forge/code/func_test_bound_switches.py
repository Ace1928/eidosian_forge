import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_bound_switches(self):
    lb = -1e-08
    ub = 1e-08
    x0 = 0.0
    jac_true = self.jac_with_nan(x0)
    jac_diff_2 = approx_derivative(self.fun_with_nan, x0, method='2-point', rel_step=1e-06, bounds=(lb, ub))
    jac_diff_3 = approx_derivative(self.fun_with_nan, x0, rel_step=1e-06, bounds=(lb, ub))
    assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
    assert_allclose(jac_diff_3, jac_true, rtol=1e-09)
    x0 = 1e-08
    jac_true = self.jac_with_nan(x0)
    jac_diff_2 = approx_derivative(self.fun_with_nan, x0, method='2-point', rel_step=1e-06, bounds=(lb, ub))
    jac_diff_3 = approx_derivative(self.fun_with_nan, x0, rel_step=1e-06, bounds=(lb, ub))
    assert_allclose(jac_diff_2, jac_true, rtol=1e-06)
    assert_allclose(jac_diff_3, jac_true, rtol=1e-09)