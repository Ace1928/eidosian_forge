import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('rng_name', ['RandomState', 'default_rng'])
def test_bootstrap_resample(rng_name):
    rng = getattr(np.random, rng_name, None)
    if rng is None:
        pytest.skip(f'{rng_name} not available.')
    rng1 = rng(0)
    rng2 = rng(0)
    n_resamples = 10
    shape = (3, 4, 5, 6)
    np.random.seed(0)
    x = np.random.rand(*shape)
    y = _resampling._bootstrap_resample(x, n_resamples, random_state=rng1)
    for i in range(n_resamples):
        slc = y[..., i, :]
        js = rng_integers(rng2, 0, shape[-1], shape[-1])
        expected = x[..., js]
        assert np.array_equal(slc, expected)