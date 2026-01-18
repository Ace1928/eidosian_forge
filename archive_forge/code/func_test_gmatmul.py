import numpy as np
import pytest
import cvxpy
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.canonicalizers as dgp_atom_canon
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest
def test_gmatmul(self) -> None:
    x = cvxpy.Variable(2, pos=True)
    A = np.array([[-5.0, 2.0], [1.0, -3.0]])
    b = np.array([3, 2])
    expr = cvxpy.gmatmul(A, x)
    x.value = b
    self.assertItemsAlmostEqual(expr.value, [3 ** (-5) * 2 ** 2, 3.0 / 8])
    A_par = cvxpy.Parameter((2, 2), value=A)
    self.assertItemsAlmostEqual(cvxpy.gmatmul(A_par, x).value, [3 ** (-5) * 2 ** 2, 3.0 / 8])
    x.value = None
    prob = cvxpy.Problem(cvxpy.Minimize(1.0), [expr == b])
    prob.solve(solver=SOLVER, gp=True)
    sltn = np.exp(np.linalg.solve(A, np.log(b)))
    self.assertItemsAlmostEqual(x.value, sltn)