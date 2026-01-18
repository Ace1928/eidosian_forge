import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_milp_4():
    c = [8, 1]
    integrality = [0, 1]
    A = [[1, 2], [-4, -1], [2, 1]]
    b_l = [-14, -np.inf, -np.inf]
    b_u = [np.inf, -33, 20]
    constraints = LinearConstraint(A, b_l, b_u)
    bounds = Bounds(-np.inf, np.inf)
    res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    assert_allclose(res.fun, 59)
    assert_allclose(res.x, [6.5, 7])