import numpy as np
from numpy.testing import assert_allclose, assert_equal  #noqa
from statsmodels.stats import weightstats
import statsmodels.stats.multivariate as smmv  # pytest cannot import test_xxx
from statsmodels.stats.multivariate import confint_mvmean_fromstats
from statsmodels.tools.testing import Holder
def test_blockdiagonal(self):
    cov, nobs = (self.cov, self.nobs)
    p_chi2 = 0.1721758850671037
    chi2 = 3.518477474111563
    block_len = [2, 1]
    stat, pv = smmv.test_cov_blockdiagonal(cov, nobs, block_len)
    assert_allclose(stat, chi2, rtol=1e-07)
    assert_allclose(pv, p_chi2, rtol=1e-06)