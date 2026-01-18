import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('a', np.arange(-2, 3))
def test_against_normaltest(self, a):
    rng = np.random.default_rng(12340513)
    x = stats.skewnorm.rvs(a=a, size=150, random_state=rng)
    expected = stats.normaltest(x)

    def statistic(x, axis):
        return stats.normaltest(x, axis=axis).statistic
    norm_rvs = self.rvs(stats.norm.rvs, rng)
    res = monte_carlo_test(x, norm_rvs, statistic, vectorized=True, alternative='greater')
    assert_allclose(res.statistic, expected.statistic)
    assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)