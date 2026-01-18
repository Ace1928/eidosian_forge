import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
def test_lse_atom(lse_example):
    x = cp.Variable(3)
    s = cp.Variable(nonneg=True)
    f_exp = cp.log_sum_exp(x)
    obj = cp.perspective(f_exp, s)
    constraints = [1 <= s, s <= 2, [1, 2, 3] <= x]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    prob.solve(solver=cp.ECOS)
    ref_prob, ref_x, ref_s = lse_example
    assert np.isclose(prob.value, ref_prob)
    assert np.allclose(x.value, ref_x)
    assert np.isclose(s.value, ref_s)