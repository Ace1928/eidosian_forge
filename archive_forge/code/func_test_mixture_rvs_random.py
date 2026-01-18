import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import (mv_mixture_rvs,
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats
def test_mixture_rvs_random(self):
    np.random.seed(0)
    mix = MixtureDistribution()
    res = mix.rvs([0.75, 0.25], 1000, dist=[stats.norm, stats.norm], kwargs=(dict(loc=-1, scale=0.5), dict(loc=1, scale=0.5)))
    npt.assert_almost_equal(np.array([res.std(), res.mean(), res.var()]), np.array([1, -0.5, 1]), decimal=1)