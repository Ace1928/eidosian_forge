import numpy as np
from copy import deepcopy
from numpy.linalg import norm
from numpy.testing import (TestCase, assert_array_almost_equal,
from scipy.optimize import (BFGS, SR1)
def test_SR1_skip_update(self):
    prob = Rosenbrock(n=5)
    x_list = [[0.097627, 0.4303787, 0.2055267, 0.0897663, -0.1526904], [0.1847239, 0.0505757, 0.2123832, 0.0255081, 0.00083286], [0.2142498, -0.018848, 0.0503822, 0.0347033, 0.03323606], [0.207168, -0.0185071, 0.0341337, -0.0139298, 0.0288175], [0.1533055, -0.0322935, 0.0280418, -0.0083592, 0.01503699], [0.1382378, -0.0276671, 0.0266161, -0.007406, 0.0280161], [0.1651957, -0.0049124, 0.0269665, -0.0040025, 0.02138184], [0.235493, 0.0443711, 0.0173959, 0.0041872, 0.00794563], [0.4168118, 0.1433867, 0.0111714, 0.0126265, -0.00658537], [0.4681972, 0.2153273, 0.0225249, 0.0152704, -0.00463809], [0.6023068, 0.3346815, 0.0731108, 0.0186618, -0.00371541], [0.6415743, 0.3985468, 0.1324422, 0.021416, -0.00062401], [0.750369, 0.5447616, 0.2804541, 0.0539851, 0.0024223], [0.7452626, 0.5644594, 0.3324679, 0.0865153, 0.0045496], [0.8059782, 0.6586838, 0.4229577, 0.145299, 0.00976702], [0.8549542, 0.7226562, 0.4991309, 0.2420093, 0.02772661], [0.8571332, 0.7285741, 0.5279076, 0.2824549, 0.06030276], [0.8835633, 0.7727077, 0.5957984, 0.3411303, 0.09652185], [0.9071558, 0.8299587, 0.67714, 0.4402896, 0.17469338]]
    grad_list = [prob.grad(x) for x in x_list]
    delta_x = [np.array(x_list[i + 1]) - np.array(x_list[i]) for i in range(len(x_list) - 1)]
    delta_grad = [grad_list[i + 1] - grad_list[i] for i in range(len(grad_list) - 1)]
    hess = SR1(init_scale=1, min_denominator=0.01)
    hess.initialize(len(x_list[0]), 'hess')
    for i in range(len(delta_x) - 1):
        s = delta_x[i]
        y = delta_grad[i]
        hess.update(s, y)
    B = np.copy(hess.get_matrix())
    s = delta_x[17]
    y = delta_grad[17]
    hess.update(s, y)
    B_updated = np.copy(hess.get_matrix())
    assert_array_equal(B, B_updated)