import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
def test_ledoit_wolf_large():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(10, 20))
    lw = LedoitWolf(block_size=10).fit(X)
    assert_almost_equal(lw.covariance_, np.eye(20), 0)
    cov = lw.covariance_
    lw = LedoitWolf(block_size=25).fit(X)
    assert_almost_equal(lw.covariance_, cov)