import re
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from .test_linprog import magic_square
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy import sparse
def test_milp_2():
    c = -np.ones(2)
    A = [[-2, 2], [-8, 10]]
    b_l = [1, -np.inf]
    b_u = [np.inf, 13]
    linear_constraint = LinearConstraint(A, b_l, b_u)
    res1 = milp(c=c, constraints=(A, b_l, b_u), integrality=True)
    res2 = milp(c=c, constraints=linear_constraint, integrality=True)
    res3 = milp(c=c, constraints=[(A, b_l, b_u)], integrality=True)
    res4 = milp(c=c, constraints=[linear_constraint], integrality=True)
    res5 = milp(c=c, integrality=True, constraints=[(A[:1], b_l[:1], b_u[:1]), (A[1:], b_l[1:], b_u[1:])])
    res6 = milp(c=c, integrality=True, constraints=[LinearConstraint(A[:1], b_l[:1], b_u[:1]), LinearConstraint(A[1:], b_l[1:], b_u[1:])])
    res7 = milp(c=c, integrality=True, constraints=[(A[:1], b_l[:1], b_u[:1]), LinearConstraint(A[1:], b_l[1:], b_u[1:])])
    xs = np.array([res1.x, res2.x, res3.x, res4.x, res5.x, res6.x, res7.x])
    funs = np.array([res1.fun, res2.fun, res3.fun, res4.fun, res5.fun, res6.fun, res7.fun])
    np.testing.assert_allclose(xs, np.broadcast_to([1, 2], xs.shape))
    np.testing.assert_allclose(funs, -3)
    res = milp(c=c, constraints=(A, b_l, b_u))
    np.testing.assert_allclose(res.x, [4, 4.5])
    np.testing.assert_allclose(res.fun, -8.5)