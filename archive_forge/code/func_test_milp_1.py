import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_milp_1():
    n = 3
    A, b, c, numbers, M = magic_square(n)
    A = sparse.csc_array(A)
    res = milp(c=c * 0, constraints=(A, b, b), bounds=(0, 1), integrality=1)
    x = np.round(res.x)
    s = (numbers.flatten() * x).reshape(n ** 2, n, n)
    square = np.sum(s, axis=0)
    np.testing.assert_allclose(square.sum(axis=0), M)
    np.testing.assert_allclose(square.sum(axis=1), M)
    np.testing.assert_allclose(np.diag(square).sum(), M)
    np.testing.assert_allclose(np.diag(square[:, ::-1]).sum(), M)