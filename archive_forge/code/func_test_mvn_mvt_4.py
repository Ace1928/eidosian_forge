import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose
from statsmodels.sandbox.distributions.multivariate import (
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal
def test_mvn_mvt_4(self):
    a, bl = (self.a, self.b)
    df = self.df
    corr2 = self.corr2
    a2 = a.copy()
    a2[:] = -np.inf
    probmvn_R = 0.1666667
    probmvt_R = 0.1666667
    assert_almost_equal(probmvt_R, mvstdtprob(np.zeros(3), -a2, corr2, df), 4)
    assert_almost_equal(probmvn_R, mvstdnormcdf(np.zeros(3), -a2, corr2, maxpts=100000, abseps=1e-05), 4)