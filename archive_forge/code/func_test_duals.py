import numpy as np
import scipy.sparse as sp
import cvxpy as cp
from cvxpy import Minimize, Problem
from cvxpy.expressions.constants import Constant, Parameter
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
def test_duals(self) -> None:
    np.random.seed(0)
    u_real = np.random.rand(3)
    u_imag = np.random.rand(3)
    u = u_real + 1j * u_imag
    y = cp.Variable(shape=(3,))
    helper_objective = cp.Minimize(cp.norm(y - u_real))
    con_a = cp.PowCone3D(y[0], y[1], y[2], [0.25])
    helper_prob_a = cp.Problem(helper_objective, [con_a])
    helper_prob_a.solve()
    expect_dual_a = con_a.dual_value
    con_b = cp.ExpCone(y[0], y[1], y[2])
    helper_prob_b = cp.Problem(helper_objective, [con_b])
    helper_prob_b.solve()
    expect_dual_b = con_b.dual_value
    con_c = cp.SOC(y[2], y[:2])
    helper_prob_c = cp.Problem(helper_objective, [con_c])
    helper_prob_c.solve()
    expect_dual_c = con_c.dual_value
    x = cp.Variable(shape=(3,), complex=True)
    actual_objective = cp.Minimize(cp.norm(x - u))
    coupling_con = cp.real(x) == y
    con_a_test = con_a.copy()
    prob_a = cp.Problem(actual_objective, [coupling_con, con_a_test])
    prob_a.solve()
    actual_dual_a = con_a_test.dual_value
    self.assertItemsAlmostEqual(actual_dual_a, expect_dual_a, places=2)
    con_b_test = con_b.copy()
    prob_b = cp.Problem(actual_objective, [coupling_con, con_b_test])
    prob_b.solve()
    actual_dual_b = con_b_test.dual_value
    self.assertItemsAlmostEqual(actual_dual_b, expect_dual_b, places=2)
    con_c_test = con_c.copy()
    prob_c = cp.Problem(actual_objective, [coupling_con, con_c_test])
    prob_c.solve()
    actual_dual_c = con_c_test.dual_value
    self.assertItemsAlmostEqual(actual_dual_c[0], expect_dual_c[0], places=2)
    self.assertItemsAlmostEqual(actual_dual_c[1], expect_dual_c[1], places=2)