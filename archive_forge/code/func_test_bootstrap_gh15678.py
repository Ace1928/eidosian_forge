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
def test_bootstrap_gh15678(method):
    rng = np.random.default_rng(354645618886684)
    dist = stats.norm(loc=2, scale=4)
    data = dist.rvs(size=100, random_state=rng)
    data = (data,)
    res = bootstrap(data, stats.skew, method=method, n_resamples=100, random_state=np.random.default_rng(9563))
    ref = bootstrap(data, stats.skew, method=method, n_resamples=100, random_state=np.random.default_rng(9563), vectorized=False)
    assert_allclose(res.confidence_interval, ref.confidence_interval)
    assert_allclose(res.standard_error, ref.standard_error)
    assert isinstance(res.standard_error, np.float64)