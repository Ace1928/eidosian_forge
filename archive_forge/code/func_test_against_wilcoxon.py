import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
def test_against_wilcoxon(self, alternative):
    x = stats.uniform.rvs(size=(3, 6, 2), loc=0, random_state=self.rng)
    y = stats.uniform.rvs(size=(3, 6, 2), loc=0.05, random_state=self.rng)

    def statistic_1samp_1d(z):
        return stats.wilcoxon(z, alternative='less').statistic

    def statistic_2samp_1d(x, y):
        return stats.wilcoxon(x, y, alternative='less').statistic

    def test_1d(x, y):
        return stats.wilcoxon(x, y, alternative=alternative)
    test = _resampling._vectorize_statistic(test_1d)
    expected = test(x, y, axis=1)
    expected_stat = expected[0]
    expected_p = expected[1]
    kwds = {'vectorized': False, 'axis': 1, 'alternative': alternative, 'permutation_type': 'samples', 'random_state': self.rng, 'n_resamples': np.inf}
    res1 = permutation_test((x - y,), statistic_1samp_1d, **kwds)
    res2 = permutation_test((x, y), statistic_2samp_1d, **kwds)
    assert_allclose(res1.statistic, res2.statistic, rtol=self.rtol)
    if alternative != 'two-sided':
        assert_allclose(res2.statistic, expected_stat, rtol=self.rtol)
    assert_allclose(res2.pvalue, expected_p, rtol=self.rtol)
    assert_allclose(res1.pvalue, res2.pvalue, rtol=self.rtol)