import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method, expected', tests_against_itself_2samp.items())
def test_bootstrap_against_itself_2samp(method, expected):
    np.random.seed(0)
    n1 = 100
    n2 = 120
    n_resamples = 999
    confidence_level = 0.9

    def my_stat(data1, data2, axis=-1):
        mean1 = np.mean(data1, axis=axis)
        mean2 = np.mean(data2, axis=axis)
        return mean1 - mean2
    dist1 = stats.norm(loc=0, scale=1)
    dist2 = stats.norm(loc=0.1, scale=1)
    stat_true = dist1.mean() - dist2.mean()
    n_replications = 1000
    data1 = dist1.rvs(size=(n_replications, n1))
    data2 = dist2.rvs(size=(n_replications, n2))
    res = bootstrap((data1, data2), statistic=my_stat, confidence_level=confidence_level, n_resamples=n_resamples, batch=50, method=method, axis=-1)
    ci = res.confidence_interval
    ci_contains_true = np.sum((ci[0] < stat_true) & (stat_true < ci[1]))
    assert ci_contains_true == expected
    pvalue = stats.binomtest(ci_contains_true, n_replications, confidence_level).pvalue
    assert pvalue > 0.1