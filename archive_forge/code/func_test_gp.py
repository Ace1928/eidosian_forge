import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def test_gp(ineq_form: bool):
    """Test FiniteSet used in a GP."""
    x = cp.Variable(pos=True)
    y = cp.Variable(pos=True)
    objective = cp.Maximize(x * y)
    set_vals = {2}
    constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form), y <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(gp=True, solver=cp.GLPK_MI)
    assert np.allclose(x.value, 2)
    assert np.allclose(y.value, 1)
    set_vals = {1, 2, 3}
    constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form), y <= 1]
    problem = cp.Problem(objective, constraints)
    problem.solve(gp=True, solver=cp.GLPK_MI)
    assert np.allclose(x.value, 3)
    assert np.allclose(y.value, 1)