import multiprocessing
import platform
from scipy.optimize._differentialevolution import (DifferentialEvolutionSolver,
from scipy.optimize import differential_evolution, OptimizeResult
from scipy.optimize._constraints import (Bounds, NonlinearConstraint,
from scipy.optimize import rosen, minimize
from scipy.sparse import csr_matrix
from scipy import stats
import numpy as np
from numpy.testing import (assert_equal, assert_allclose, assert_almost_equal,
from pytest import raises as assert_raises, warns
import pytest
def test_L7(self):

    def f(x):
        x = np.hstack(([0], x))
        fun = 5.3578547 * x[3] ** 2 + 0.8356891 * x[1] * x[5] + 37.293239 * x[1] - 40792.141
        return fun

    def c1(x):
        x = np.hstack(([0], x))
        return [85.334407 + 0.0056858 * x[2] * x[5] + 0.0006262 * x[1] * x[4] - 0.0022053 * x[3] * x[5], 80.51249 + 0.0071317 * x[2] * x[5] + 0.0029955 * x[1] * x[2] + 0.0021813 * x[3] ** 2, 9.300961 + 0.0047026 * x[3] * x[5] + 0.0012547 * x[1] * x[3] + 0.0019085 * x[3] * x[4]]
    N = NonlinearConstraint(c1, [0, 90, 20], [92, 110, 25])
    bounds = [(78, 102), (33, 45)] + [(27, 45)] * 3
    constraints = N
    res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234, constraints=constraints)
    x_opt = [78.00000686, 33.00000362, 29.99526064, 44.99999971, 36.77579979]
    f_opt = -30665.537578
    assert_allclose(f(x_opt), f_opt)
    assert_allclose(res.x, x_opt, atol=0.001)
    assert_allclose(res.fun, f_opt, atol=0.001)
    assert res.success
    assert_(np.all(np.array(c1(res.x)) >= np.array([0, 90, 20])))
    assert_(np.all(np.array(c1(res.x)) <= np.array([92, 110, 25])))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))