import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method, expected', tests_R.items())
def test_bootstrap_against_R(method, expected):
    x = np.array([10, 12, 12.5, 12.5, 13.9, 15, 21, 22, 23, 34, 50, 81, 89, 121, 134, 213])
    res = bootstrap((x,), np.mean, n_resamples=1000000, method=method, random_state=0)
    assert_allclose(res.confidence_interval, expected, rtol=0.005)