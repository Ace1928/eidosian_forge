import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_against_f_oneway(self):
    rng = np.random.default_rng(219017667302737545)
    data = (rng.random(size=(2, 100)), rng.random(size=(2, 101)), rng.random(size=(2, 102)), rng.random(size=(2, 103)))
    rvs = (rng.normal, rng.normal, rng.normal, rng.normal)

    def statistic(*args, axis):
        return stats.f_oneway(*args, axis=axis).statistic
    res = stats.monte_carlo_test(data, rvs, statistic, axis=-1, alternative='greater')
    ref = stats.f_oneway(*data, axis=-1)
    assert_allclose(res.statistic, ref.statistic)
    assert_allclose(res.pvalue, ref.pvalue, atol=0.01)