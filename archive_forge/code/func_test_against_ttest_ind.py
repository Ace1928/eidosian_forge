import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_against_ttest_ind(self):
    rng = np.random.default_rng(219017667302737545)
    data = (rng.random(size=(2, 5)), rng.random(size=7))
    rvs = (rng.normal, rng.normal)

    def statistic(x, y, axis):
        return stats.ttest_ind(x, y, axis).statistic
    res = stats.monte_carlo_test(data, rvs, statistic, axis=-1)
    ref = stats.ttest_ind(data[0], [data[1]], axis=-1)
    assert_allclose(res.statistic, ref.statistic)
    assert_allclose(res.pvalue, ref.pvalue, rtol=0.02)