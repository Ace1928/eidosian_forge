import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_infeasible_prob_16609():
    c = [1.0, 0.0]
    integrality = [0, 1]
    lb = [0, -np.inf]
    ub = [np.inf, np.inf]
    bounds = Bounds(lb, ub)
    A_eq = [[0.0, 1.0]]
    b_eq = [0.5]
    constraints = LinearConstraint(A_eq, b_eq, b_eq)
    res = milp(c, integrality=integrality, bounds=bounds, constraints=constraints)
    np.testing.assert_equal(res.status, 2)