from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_chebyshev_center(self) -> None:
    a1 = np.array([2, 1])
    a2 = np.array([2, -1])
    a3 = np.array([-1, 2])
    a4 = np.array([-1, -2])
    b = np.ones(4)
    r = cvx.Variable(name='r')
    x_c = cvx.Variable(2, name='x_c')
    obj = cvx.Maximize(r)
    constraints = [a1.T @ x_c + np.linalg.norm(a1) * r <= b[0], a2.T @ x_c + np.linalg.norm(a2) * r <= b[1], a3.T @ x_c + np.linalg.norm(a3) * r <= b[2], a4.T @ x_c + np.linalg.norm(a4) * r <= b[3]]
    p = cvx.Problem(obj, constraints)
    result = p.solve(solver=cvx.SCS)
    self.assertAlmostEqual(result, 0.447214)
    self.assertAlmostEqual(r.value, result)
    self.assertItemsAlmostEqual(x_c.value, [0, 0])