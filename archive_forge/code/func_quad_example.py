import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
@pytest.fixture(params=[2, 3, 4, -2, 0])
def quad_example(request):
    x = cp.Variable()
    s = cp.Variable()
    r = request.param
    obj = cp.quad_over_lin(x, s) + r * x - 4 * s
    constraints = [x >= 2, s <= 0.5]
    prob_ref = cp.Problem(cp.Minimize(obj), constraints)
    prob_ref.solve(solver=cp.ECOS)
    return (prob_ref.value, s.value, x.value, r)