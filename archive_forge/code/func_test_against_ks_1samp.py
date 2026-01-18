import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('alternative', ('less', 'greater'))
@pytest.mark.parametrize('a', np.linspace(-0.5, 0.5, 5))
def test_against_ks_1samp(self, alternative, a):
    rng = np.random.default_rng(65723433)
    x = stats.skewnorm.rvs(a=a, size=30, random_state=rng)
    expected = stats.ks_1samp(x, stats.norm.cdf, alternative=alternative)

    def statistic1d(x):
        return stats.ks_1samp(x, stats.norm.cdf, mode='asymp', alternative=alternative).statistic
    norm_rvs = self.rvs(stats.norm.rvs, rng)
    res = monte_carlo_test(x, norm_rvs, statistic1d, n_resamples=1000, vectorized=False, alternative=alternative)
    assert_allclose(res.statistic, expected.statistic)
    if alternative == 'greater':
        assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)
    elif alternative == 'less':
        assert_allclose(1 - res.pvalue, expected.pvalue, atol=self.atol)