import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def test_monomial(ineq_form: bool):
    """Test FiniteSet applied to a monomial."""
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    objective = cp.Maximize(x * y)
    set_vals = {1, 2, 3}
    constraints = [FiniteSet(x * y, set_vals, ineq_form=ineq_form), x == 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(gp=True, solver=cp.GLPK_MI, ignore_dpp=True)
    assert np.allclose(x.value, 1)
    assert np.allclose(y.value, 3)
    problem.solve(gp=True, solver=cp.GLPK_MI, enforce_dpp=True)
    assert np.allclose(x.value, 1)
    assert np.allclose(y.value, 3)