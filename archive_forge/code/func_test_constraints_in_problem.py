import copy
import numpy as np
import cvxpy as cp
from cvxpy.constraints import Equality
def test_constraints_in_problem():
    x = cp.Variable(name='x', nonneg=True)
    y = cp.Variable(name='y', nonneg=True)
    original_constraints = [x + y == 1]
    shallow_constraints = copy.copy(original_constraints)
    obj = cp.Maximize(x + 2 * y)
    prob = cp.Problem(obj, shallow_constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.allclose(x.value, 0)
    assert np.allclose(y.value, 1)
    deep_constraints = copy.deepcopy(original_constraints)
    prob = cp.Problem(obj, deep_constraints)
    prob.solve()
    assert prob.status == cp.UNBOUNDED
    x_copied = deep_constraints[0].variables()[0]
    y_copied = deep_constraints[0].variables()[1]
    deep_obj = cp.Maximize(x_copied + 2 * y_copied)
    prob = cp.Problem(deep_obj, deep_constraints)
    prob.solve()
    assert prob.status == cp.OPTIMAL
    assert np.allclose(x_copied.value, 0)
    assert np.allclose(y_copied.value, 1)