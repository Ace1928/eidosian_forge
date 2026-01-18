import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (BFGS, SR1)
def test_BFGS_skip_update(self):
    prob = Rosenbrock(n=5)
    x_list = [[0.097627, 0.4303787, 0.2055267, 0.0897663, -0.1526904], [0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286], [0.2142498, -0.018848, 0.0503822, 0.0347033, 0.03323606], [0.207168, -0.0185071, 0.0341337, -0.0139298, 0.0288175], [0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699], [0.1382378, -0.0276671, 0.0266161, -0.007406, 0.0280161], [0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184]]
    grad_list = [prob.grad(x) for x in x_list]
    delta_x = [np.array(x_list[i + 1]) - np.array(x_list[i]) for i in range(len(x_list) - 1)]
    delta_grad = [grad_list[i + 1] - grad_list[i] for i in range(len(grad_list) - 1)]
    hess = BFGS(init_scale=1, min_curvature=10)
    hess.initialize(len(x_list[0]), 'hess')
    for i in range(len(delta_x) - 1):
        s = delta_x[i]
        y = delta_grad[i]
        hess.update(s, y)
    B = np.copy(hess.get_matrix())
    s = delta_x[5]
    y = delta_grad[5]
    hess.update(s, y)
    B_updated = np.copy(hess.get_matrix())
    assert_array_equal(B, B_updated)