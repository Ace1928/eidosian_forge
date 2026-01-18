import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from scipy import stats
from scipy.stats import _survival
def test_evaluate_methods(self):
    rng = np.random.default_rng(1162729143302572461)
    sample, _, _ = self.get_random_sample(rng, 15)
    res = stats.ecdf(sample)
    x = res.cdf.quantiles
    xr = x + np.diff(x, append=x[-1] + 1) / 2
    assert_equal(res.cdf.evaluate(x), res.cdf.probabilities)
    assert_equal(res.cdf.evaluate(xr), res.cdf.probabilities)
    assert_equal(res.cdf.evaluate(x[0] - 1), 0)
    assert_equal(res.cdf.evaluate([-np.inf, np.inf]), [0, 1])
    assert_equal(res.sf.evaluate(x), res.sf.probabilities)
    assert_equal(res.sf.evaluate(xr), res.sf.probabilities)
    assert_equal(res.sf.evaluate(x[0] - 1), 1)
    assert_equal(res.sf.evaluate([-np.inf, np.inf]), [1, 0])