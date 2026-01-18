import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
def test_ledoit_wolf_small():
    X_small = X[:, :4]
    lw = LedoitWolf()
    lw.fit(X_small)
    shrinkage_ = lw.shrinkage_
    assert_almost_equal(shrinkage_, _naive_ledoit_wolf_shrinkage(X_small))