import numpy as np
from numpy.testing import assert_almost_equal,  assert_allclose
from statsmodels.sandbox.distributions.multivariate import (
from statsmodels.sandbox.distributions.mv_normal import MVT, MVNormal
def test_mvn_mvt_3(self):
    a, b = (self.a, self.b)
    df = self.df
    corr2 = self.corr2
    a2 = a.copy()
    a2[:] = -np.inf
    probmvn_R = 0.9961141
    probmvt_R = 0.9522146
    quadkwds = {'epsabs': 1e-08}
    probmvt = mvstdtprob(a2, b, corr2, df, quadkwds=quadkwds)
    assert_allclose(probmvt_R, probmvt, atol=0.0005)
    probmvn = mvstdnormcdf(a2, b, corr2, maxpts=100000, abseps=1e-05)
    assert_allclose(probmvn_R, probmvn, atol=0.0001)