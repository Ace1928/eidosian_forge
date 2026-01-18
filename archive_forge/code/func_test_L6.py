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
def test_L6(self):

    def f(x):
        x = np.hstack(([0], x))
        fun = (x[1] - 10) ** 3 + (x[2] - 20) ** 3
        return fun

    def c1(x):
        x = np.hstack(([0], x))
        return [(x[1] - 5) ** 2 + (x[2] - 5) ** 2 - 100, -(x[1] - 6) ** 2 - (x[2] - 5) ** 2 + 82.81]
    N = NonlinearConstraint(c1, 0, np.inf)
    bounds = [(13, 100), (0, 100)]
    constraints = N
    res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234, constraints=constraints, tol=1e-07)
    x_opt = (14.095, 0.84296)
    f_opt = -6961.814744
    assert_allclose(f(x_opt), f_opt, atol=1e-06)
    assert_allclose(res.fun, f_opt, atol=0.001)
    assert_allclose(res.x, x_opt, atol=0.0001)
    assert res.success
    assert_(np.all(np.array(c1(res.x)) >= 0))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))