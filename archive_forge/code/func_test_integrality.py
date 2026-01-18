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
def test_integrality(self):
    rng = np.random.default_rng(6519843218105)
    dist = stats.nbinom
    shapes = (5, 0.5)
    x = dist.rvs(*shapes, size=10000, random_state=rng)

    def func(p, *args):
        dist, x = args
        ll = -np.log(dist.pmf(x, *p)).sum(axis=-1)
        if np.isnan(ll):
            ll = np.inf
        return ll
    integrality = [True, False]
    bounds = [(1, 18), (0, 0.95)]
    res = differential_evolution(func, bounds, args=(dist, x), integrality=integrality, polish=False, seed=rng)
    assert res.x[0] == 5
    assert_allclose(res.x, shapes, rtol=0.025)
    res2 = differential_evolution(func, bounds, args=(dist, x), integrality=integrality, polish=True, seed=rng)

    def func2(p, *args):
        n, dist, x = args
        return func(np.array([n, p[0]]), dist, x)
    LBFGSB = minimize(func2, res2.x[1], args=(5, dist, x), bounds=[(0, 0.95)])
    assert_allclose(res2.x[1], LBFGSB.x)
    assert res2.fun <= res.fun