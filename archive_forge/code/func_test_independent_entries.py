import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@staticmethod
def test_independent_entries(ineq_form: bool):
    shape = (2, 2)
    x = cp.Variable(shape)
    objective = cp.Maximize(cp.sum(x))
    set_vals = {0, 1, 2}
    constraints = [FiniteSet(x, set_vals, ineq_form=ineq_form), x <= np.arange(4).reshape(shape)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    assert np.allclose(x.value, np.array([[0, 1], [2, 2]]))