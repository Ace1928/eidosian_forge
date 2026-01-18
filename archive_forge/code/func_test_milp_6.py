import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
@pytest.mark.slow
@pytest.mark.timeout(120)
def test_milp_6():
    integrality = 1
    A_eq = np.array([[22, 13, 26, 33, 21, 3, 14, 26], [39, 16, 22, 28, 26, 30, 23, 24], [18, 14, 29, 27, 30, 38, 26, 26], [41, 26, 28, 36, 18, 38, 16, 26]])
    b_eq = np.array([7872, 10466, 11322, 12058])
    c = np.array([2, 10, 13, 17, 7, 5, 7, 3])
    res = milp(c=c, constraints=(A_eq, b_eq, b_eq), integrality=integrality)
    np.testing.assert_allclose(res.fun, 1854)