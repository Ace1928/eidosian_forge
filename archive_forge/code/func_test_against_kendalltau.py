import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_against_kendalltau(self):
    x = self.rng.normal(size=6)
    y = x + self.rng.normal(size=6)
    expected = stats.kendalltau(x, y, method='exact')

    def statistic1d(x):
        return stats.kendalltau(x, y, method='asymptotic').statistic
    res = permutation_test((x,), statistic1d, permutation_type='pairings', n_resamples=np.inf, random_state=self.rng)
    assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)