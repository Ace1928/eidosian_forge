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
def test_L2(self):

    def f(x):
        x = np.hstack(([0], x))
        fun = (x[1] - 10) ** 2 + 5 * (x[2] - 12) ** 2 + x[3] ** 4 + 3 * (x[4] - 11) ** 2 + 10 * x[5] ** 6 + 7 * x[6] ** 2 + x[7] ** 4 - 4 * x[6] * x[7] - 10 * x[6] - 8 * x[7]
        return fun

    def c1(x):
        x = np.hstack(([0], x))
        return [127 - 2 * x[1] ** 2 - 3 * x[2] ** 4 - x[3] - 4 * x[4] ** 2 - 5 * x[5], 196 - 23 * x[1] - x[2] ** 2 - 6 * x[6] ** 2 + 8 * x[7], 282 - 7 * x[1] - 3 * x[2] - 10 * x[3] ** 2 - x[4] + x[5], -4 * x[1] ** 2 - x[2] ** 2 + 3 * x[1] * x[2] - 2 * x[3] ** 2 - 5 * x[6] + 11 * x[7]]
    N = NonlinearConstraint(c1, 0, np.inf)
    bounds = [(-10, 10)] * 7
    constraints = N
    with suppress_warnings() as sup:
        sup.filter(UserWarning)
        res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234, constraints=constraints)
    f_opt = 680.6300599487869
    x_opt = (2.330499, 1.951372, -0.4775414, 4.365726, -0.624487, 1.038131, 1.594227)
    assert_allclose(f(x_opt), f_opt)
    assert_allclose(res.fun, f_opt)
    assert_allclose(res.x, x_opt, atol=1e-05)
    assert res.success
    assert_(np.all(np.array(c1(res.x)) >= 0))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))