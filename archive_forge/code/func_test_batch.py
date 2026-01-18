import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('random_state', [np.random.RandomState, np.random.default_rng])
@pytest.mark.parametrize('permutation_type', ['pairings', 'samples', 'independent'])
def test_batch(self, permutation_type, random_state):
    x = self.rng.random(10)
    y = self.rng.random(10)

    def statistic(x, y, axis):
        batch_size = 1 if x.ndim == 1 else len(x)
        statistic.batch_size = max(batch_size, statistic.batch_size)
        statistic.counter += 1
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)
    statistic.counter = 0
    statistic.batch_size = 0
    kwds = {'n_resamples': 1000, 'permutation_type': permutation_type, 'vectorized': True}
    res1 = stats.permutation_test((x, y), statistic, batch=1, random_state=random_state(0), **kwds)
    assert_equal(statistic.counter, 1001)
    assert_equal(statistic.batch_size, 1)
    statistic.counter = 0
    res2 = stats.permutation_test((x, y), statistic, batch=50, random_state=random_state(0), **kwds)
    assert_equal(statistic.counter, 21)
    assert_equal(statistic.batch_size, 50)
    statistic.counter = 0
    res3 = stats.permutation_test((x, y), statistic, batch=1000, random_state=random_state(0), **kwds)
    assert_equal(statistic.counter, 2)
    assert_equal(statistic.batch_size, 1000)
    assert_equal(res1.pvalue, res3.pvalue)
    assert_equal(res2.pvalue, res3.pvalue)