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
def test_L9(self):

    def f(x):
        x = np.hstack(([0], x))
        return x[1] ** 2 + (x[2] - 1) ** 2

    def c1(x):
        x = np.hstack(([0], x))
        return [x[2] - x[1] ** 2]
    N = NonlinearConstraint(c1, [-0.001], [0.001])
    bounds = [(-1, 1)] * 2
    constraints = N
    res = differential_evolution(f, bounds, strategy='rand1bin', seed=1234, constraints=constraints)
    x_opt = [np.sqrt(2) / 2, 0.5]
    f_opt = 0.75
    assert_allclose(f(x_opt), f_opt)
    assert_allclose(np.abs(res.x), x_opt, atol=0.001)
    assert_allclose(res.fun, f_opt, atol=0.001)
    assert res.success
    assert_(np.all(np.array(c1(res.x)) >= -0.001))
    assert_(np.all(np.array(c1(res.x)) <= 0.001))
    assert_(np.all(res.x >= np.array(bounds)[:, 0]))
    assert_(np.all(res.x <= np.array(bounds)[:, 1]))