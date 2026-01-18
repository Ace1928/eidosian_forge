import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
def test_fast_mcd_on_invalid_input():
    X = np.arange(100)
    msg = 'Expected 2D array, got 1D array instead'
    with pytest.raises(ValueError, match=msg):
        fast_mcd(X)