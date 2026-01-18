import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pytest
from statsmodels.tools.eval_measures import (
def test_ic():
    n = 10
    k = 2
    assert_almost_equal(aic(0, 10, 2), 2 * k, decimal=14)
    assert_almost_equal(aicc(0, 10, 2), aic(0, n, k) + 2 * k * (k + 1.0) / (n - k - 1.0), decimal=14)
    assert_almost_equal(bic(0, 10, 2), np.log(n) * k, decimal=14)
    assert_almost_equal(hqic(0, 10, 2), 2 * np.log(np.log(n)) * k, decimal=14)