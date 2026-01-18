import unittest
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
from cvxpy.tests.base_test import BaseTest
def bool_prob(self, solver) -> None:
    obj = cp.Minimize(cp.abs(self.x_bool - 0.2))
    p = cp.Problem(obj, [])
    result = p.solve(solver=solver)
    self.assertAlmostEqual(result, 0.2)
    self.assertAlmostEqual(self.x_bool.value, 0)
    t = cp.Variable()
    obj = cp.Minimize(t)
    p = cp.Problem(obj, [cp.abs(self.x_bool) <= t])
    result = p.solve(solver=solver)
    self.assertAlmostEqual(result, 0)
    self.assertAlmostEqual(self.x_bool.value, 0, places=4)
    C = np.array([[0, 1, 0], [1, 1, 1]]).T
    obj = cp.Minimize(cp.sum(cp.abs(self.A_bool - C)))
    p = cp.Problem(obj, [])
    result = p.solve(solver=solver)
    self.assertAlmostEqual(result, 0)
    self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)
    t = cp.Variable()
    obj = cp.Minimize(t)
    p = cp.Problem(obj, [cp.sum(cp.abs(self.A_bool - C)) <= t])
    result = p.solve(solver=solver)
    self.assertAlmostEqual(result, 0)
    self.assertItemsAlmostEqual(self.A_bool.value, C, places=4)