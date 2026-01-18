import sys
import numpy as np
from numpy.testing import (assert_, assert_array_equal, assert_allclose,
from pytest import raises as assert_raises
from scipy.sparse import coo_matrix
from scipy.special import erf
from scipy.integrate._bvp import (modify_mesh, estimate_fun_jac,
def test_big_problem_with_parameters():
    n = 30
    x = np.linspace(0, np.pi, 5)
    x_test = np.linspace(0, np.pi, 100)
    y = np.ones((2 * n, x.size))
    for fun_jac in [None, big_fun_with_parameters_jac]:
        for bc_jac in [None, big_bc_with_parameters_jac]:
            sol = solve_bvp(big_fun_with_parameters, big_bc_with_parameters, x, y, p=[0.5, 0.5], fun_jac=fun_jac, bc_jac=bc_jac)
            assert_equal(sol.status, 0)
            assert_(sol.success)
            assert_allclose(sol.p, [1, 1], rtol=0.0001)
            sol_test = sol.sol(x_test)
            for isol in range(0, n, 4):
                assert_allclose(sol_test[isol], big_sol_with_parameters(x_test, [1, 1])[0], rtol=0.0001, atol=0.0001)
                assert_allclose(sol_test[isol + 2], big_sol_with_parameters(x_test, [1, 1])[1], rtol=0.0001, atol=0.0001)
            f_test = big_fun_with_parameters(x_test, sol_test, [1, 1])
            r = sol.sol(x_test, 1) - f_test
            rel_res = r / (1 + np.abs(f_test))
            norm_res = np.sum(rel_res ** 2, axis=0) ** 0.5
            assert_(np.all(norm_res < 0.001))
            assert_(np.all(sol.rms_residuals < 0.001))
            assert_allclose(sol.sol(sol.x), sol.y, rtol=1e-10, atol=1e-10)
            assert_allclose(sol.sol(sol.x, 1), sol.yp, rtol=1e-10, atol=1e-10)