import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method, expected', tests_against_itself_1samp.items())
def test_bootstrap_against_itself_1samp(method, expected):
    np.random.seed(0)
    n = 100
    n_resamples = 999
    confidence_level = 0.9
    dist = stats.norm(loc=5, scale=1)
    stat_true = dist.mean()
    n_replications = 2000
    data = dist.rvs(size=(n_replications, n))
    res = bootstrap((data,), statistic=np.mean, confidence_level=confidence_level, n_resamples=n_resamples, batch=50, method=method, axis=-1)
    ci = res.confidence_interval
    ci_contains_true = np.sum((ci[0] < stat_true) & (stat_true < ci[1]))
    assert ci_contains_true == expected
    pvalue = stats.binomtest(ci_contains_true, n_replications, confidence_level).pvalue
    assert pvalue > 0.1