import numpy as np
import pytest
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests import solver_test_helpers as STH
@solver_installed
def test_default_argument():
    x = cp.Variable()
    objective = cp.Maximize(x)
    set_vals = set(range(5))
    constraints = [FiniteSet(x, set_vals)]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.GLPK_MI)
    assert np.allclose(x.value, 4)