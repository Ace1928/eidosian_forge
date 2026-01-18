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
def test_against_ks_2samp(self, alternative):
    x = self.rng.normal(size=4, scale=1)
    y = self.rng.normal(size=5, loc=3, scale=3)
    expected = stats.ks_2samp(x, y, alternative=alternative, mode='exact')

    def statistic1d(x, y):
        return stats.ks_2samp(x, y, mode='asymp', alternative=alternative).statistic
    res = permutation_test((x, y), statistic1d, n_resamples=np.inf, alternative='greater', random_state=self.rng)
    assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)