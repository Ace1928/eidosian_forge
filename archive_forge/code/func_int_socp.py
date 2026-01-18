import unittest
import numpy as np
import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as MIP_SOLVERS
from cvxpy.tests.base_test import BaseTest
def int_socp(self, solver) -> None:
    t = cp.Variable()
    obj = cp.Minimize(t)
    p = cp.Problem(obj, [cp.square(self.y_int - 0.2) <= t])
    result = p.solve(solver=solver)
    self.assertAlmostEqual(result, 0.04)
    self.assertAlmostEqual(self.y_int.value, 0)