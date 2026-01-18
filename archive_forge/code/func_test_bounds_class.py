from scipy.optimize import dual_annealing, Bounds
from scipy.optimize._dual_annealing import EnergyState
from scipy.optimize._dual_annealing import LocalSearchWrapper
from scipy.optimize._dual_annealing import ObjectiveFunWrapper
from scipy.optimize._dual_annealing import StrategyChain
from scipy.optimize._dual_annealing import VisitingDistribution
from scipy.optimize import rosen, rosen_der
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_array_less
from pytest import raises as assert_raises
from scipy._lib._util import check_random_state
def test_bounds_class(self):

    def func(x):
        f = np.sum(x * x - 10 * np.cos(2 * np.pi * x)) + 10 * np.size(x)
        return f
    lw = [-5.12] * 5
    up = [5.12] * 5
    up[0] = -2.0
    up[1] = -1.0
    lw[3] = 1.0
    lw[4] = 2.0
    bounds = Bounds(lw, up)
    ret_bounds_class = dual_annealing(func, bounds=bounds, seed=1234)
    bounds_old = list(zip(lw, up))
    ret_bounds_list = dual_annealing(func, bounds=bounds_old, seed=1234)
    assert_allclose(ret_bounds_class.x, ret_bounds_list.x, atol=1e-08)
    assert_allclose(ret_bounds_class.x, np.arange(-2, 3), atol=1e-07)
    assert_allclose(ret_bounds_list.fun, ret_bounds_class.fun, atol=1e-09)
    assert ret_bounds_list.nfev == ret_bounds_class.nfev