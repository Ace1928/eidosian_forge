import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import (mv_mixture_rvs,
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats
def test_mixture_cdf(self):
    mix = MixtureDistribution()
    grid = np.linspace(-4, 4, 10)
    res = mix.cdf(grid, [1 / 3.0, 2 / 3.0], dist=[stats.norm, stats.norm], kwargs=(dict(loc=-1, scale=0.25), dict(loc=1, scale=0.75)))
    npt.assert_almost_equal(res, np.array([8.72261646e-12, 1.4059296e-08, 5.95819161e-06, 0.0310250226, 0.346993159, 0.486283549, 0.781092904, 0.965606734, 0.998373155, 0.999978886]))