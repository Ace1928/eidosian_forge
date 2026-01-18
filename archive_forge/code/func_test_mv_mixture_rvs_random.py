import numpy as np
import numpy.testing as npt
from statsmodels.distributions.mixture_rvs import (mv_mixture_rvs,
import statsmodels.sandbox.distributions.mv_normal as mvd
from scipy import stats
def test_mv_mixture_rvs_random(self):
    cov3 = np.array([[1.0, 0.5, 0.75], [0.5, 1.5, 0.6], [0.75, 0.6, 2.0]])
    mu = np.array([-1, 0.0, 2.0])
    mu2 = np.array([4, 2.0, 2.0])
    mvn3 = mvd.MVNormal(mu, cov3)
    mvn32 = mvd.MVNormal(mu2, cov3 / 2.0)
    np.random.seed(0)
    res = mv_mixture_rvs([0.4, 0.6], 5000, [mvn3, mvn32], 3)
    npt.assert_almost_equal(np.array([res.std(), res.mean(), res.var()]), np.array([1.874, 1.733, 3.512]), decimal=1)