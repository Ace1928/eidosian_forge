import numpy as np
import pytest
import cvxpy as cp
from cvxpy.atoms.perspective import perspective
from cvxpy.constraints.exponential import ExpCone
@pytest.mark.parametrize('x_val,s_val', [(1, 2), (5, 0.25), (0.5, 7)])
def test_evaluate_persp(x_val, s_val):
    x = cp.Variable()
    s = cp.Variable(nonneg=True)
    f_exp = cp.square(x) + 3 * x - 5
    obj = cp.perspective(f_exp, s)
    val_array = np.array([s_val, x_val])
    x.value = np.array(x_val)
    s.value = np.array(s_val)
    val = obj.numeric(val_array)
    ref_val = x_val ** 2 / s_val + 3 * x_val - 5 * s_val
    assert np.isclose(val, ref_val)