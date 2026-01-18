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
def test_against_binomtest(self, alternative):
    x = self.rng.integers(0, 2, size=10)
    x[x == 0] = -1

    def statistic(x, axis=0):
        return np.sum(x > 0, axis=axis)
    k, n, p = (statistic(x), 10, 0.5)
    expected = stats.binomtest(k, n, p, alternative=alternative)
    res = stats.permutation_test((x,), statistic, vectorized=True, permutation_type='samples', n_resamples=np.inf, random_state=self.rng, alternative=alternative)
    assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)