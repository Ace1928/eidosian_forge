import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method', ['basic', 'percentile', 'BCa'])
def test_bootstrap_paired(method):
    np.random.seed(0)
    n = 100
    x = np.random.rand(n)
    y = np.random.rand(n)

    def my_statistic(x, y, axis=-1):
        return ((x - y) ** 2).mean(axis=axis)

    def my_paired_statistic(i, axis=-1):
        a = x[i]
        b = y[i]
        res = my_statistic(a, b)
        return res
    i = np.arange(len(x))
    res1 = bootstrap((i,), my_paired_statistic, random_state=0)
    res2 = bootstrap((x, y), my_statistic, paired=True, random_state=0)
    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    assert_allclose(res1.standard_error, res2.standard_error)