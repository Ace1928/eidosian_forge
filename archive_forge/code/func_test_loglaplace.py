import numpy as np
from numpy.testing import assert_almost_equal
from scipy import stats
from statsmodels.sandbox.distributions.extras import (
def test_loglaplace():
    loglaplaceexpg = ExpTransf_gen(stats.laplace)
    cdfst = stats.loglaplace.cdf(3, 3)
    cdftr = loglaplaceexpg._cdf(3, 0, 1.0 / 3)
    assert_almost_equal(cdfst, cdftr, 14)