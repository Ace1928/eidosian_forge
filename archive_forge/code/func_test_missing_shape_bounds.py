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
def test_missing_shape_bounds(self):
    N = 1000
    rng = np.random.default_rng(self.seed)
    dist = stats.binom
    n, p, loc = (10, 0.65, 0)
    data = dist.rvs(n, p, loc=loc, size=N, random_state=rng)
    shape_bounds = {'n': np.array([0, 20])}
    res = stats.fit(dist, data, shape_bounds, optimizer=self.opt)
    assert_allclose(res.params, (n, p, loc), **self.tols)
    dist = stats.bernoulli
    p, loc = (0.314159, 0)
    data = dist.rvs(p, loc=loc, size=N, random_state=rng)
    res = stats.fit(dist, data, optimizer=self.opt)
    assert_allclose(res.params, (p, loc), **self.tols)