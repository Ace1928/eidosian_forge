import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_against_cvm(self):
    x = stats.norm.rvs(size=4, scale=1, random_state=self.rng)
    y = stats.norm.rvs(size=5, loc=3, scale=3, random_state=self.rng)
    expected = stats.cramervonmises_2samp(x, y, method='exact')

    def statistic1d(x, y):
        return stats.cramervonmises_2samp(x, y, method='asymptotic').statistic
    res = permutation_test((x, y), statistic1d, n_resamples=np.inf, alternative='greater', random_state=self.rng)
    assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)