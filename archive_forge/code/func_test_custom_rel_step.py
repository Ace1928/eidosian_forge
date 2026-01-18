import math
from itertools import product
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
from pytest import raises as assert_raises
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.optimize._numdiff import (
def test_custom_rel_step(self):
    x0 = np.array([-0.1, 0.1])
    jac_diff_2 = approx_derivative(self.fun_vector_vector, x0, method='2-point', rel_step=0.0001)
    jac_diff_3 = approx_derivative(self.fun_vector_vector, x0, rel_step=0.0001)
    jac_true = self.jac_vector_vector(x0)
    assert_allclose(jac_diff_2, jac_true, rtol=0.01)
    assert_allclose(jac_diff_3, jac_true, rtol=0.0001)