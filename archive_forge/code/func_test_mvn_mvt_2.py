import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose
from statsmodels.sandbox.distributions.multivariate import (
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal
def test_mvn_mvt_2(self):
    a, b = (self.a, self.b)
    df = self.df
    corr2 = self.corr2
    probmvn_R = 0.6472497
    probmvt_R = 0.5881863
    assert_almost_equal(probmvt_R, mvstdtprob(a, b, corr2, df), 4)
    assert_almost_equal(probmvn_R, mvstdnormcdf(a, b, corr2, abseps=1e-05), 4)