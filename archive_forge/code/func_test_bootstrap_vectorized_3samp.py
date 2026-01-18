import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method', ['basic', 'percentile'])
@pytest.mark.parametrize('axis', [0, 1])
def test_bootstrap_vectorized_3samp(method, axis):

    def statistic(*data, axis=0):
        return sum((sample.mean(axis) for sample in data))

    def statistic_1d(*data):
        for sample in data:
            assert sample.ndim == 1
        return statistic(*data, axis=0)
    np.random.seed(0)
    x = np.random.rand(4, 5)
    y = np.random.rand(4, 5)
    z = np.random.rand(4, 5)
    res1 = bootstrap((x, y, z), statistic, vectorized=True, axis=axis, n_resamples=100, method=method, random_state=0)
    res2 = bootstrap((x, y, z), statistic_1d, vectorized=False, axis=axis, n_resamples=100, method=method, random_state=0)
    assert_allclose(res1.confidence_interval, res2.confidence_interval)
    assert_allclose(res1.standard_error, res2.standard_error)