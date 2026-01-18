import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose
from statsmodels.sandbox.distributions.multivariate import (
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal
def test_mvn_mvt_1(self):
    a, b = (self.a, self.b)
    df = self.df
    corr_equal = self.corr_equal
    probmvt_R = 0.60414
    probmvn_R = 0.67397
    assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr_equal, df), 4)
    assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr_equal, abseps=1e-05), 4)
    mvn_high = mvstdnormcdf(a, b, corr_equal, abseps=1e-08, maxpts=10000000)
    assert_almost_equal(probmvn_R, mvn_high, 5)