import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def test_everything_fixed(self):
    N = 5000
    rng = np.random.default_rng(self.seed)
    dist = stats.norm
    loc, scale = (1.5, 2.5)
    data = dist.rvs(loc=loc, scale=scale, size=N, random_state=rng)
    res = stats.fit(dist, data)
    assert_allclose(res.params, (0, 1), **self.tols)
    bounds = {'loc': (loc, loc), 'scale': (scale, scale)}
    res = stats.fit(dist, data, bounds)
    assert_allclose(res.params, (loc, scale), **self.tols)
    dist = stats.binom
    n, p, loc = (10, 0.65, 0)
    data = dist.rvs(n, p, loc=loc, size=N, random_state=rng)
    shape_bounds = {'n': (0, 20), 'p': (0.65, 0.65)}
    res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
    assert_allclose(res.params, (n, p, loc), **self.tols)