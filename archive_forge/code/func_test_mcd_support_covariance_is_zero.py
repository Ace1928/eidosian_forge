import itertools
import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import MinCovDet, empirical_covariance, fast_mcd
from sklearn.utils._testing import assert_array_almost_equal
def test_mcd_support_covariance_is_zero():
    X_1 = np.array([0.5, 0.1, 0.1, 0.1, 0.957, 0.1, 0.1, 0.1, 0.4285, 0.1])
    X_1 = X_1.reshape(-1, 1)
    X_2 = np.array([0.5, 0.3, 0.3, 0.3, 0.957, 0.3, 0.3, 0.3, 0.4285, 0.3])
    X_2 = X_2.reshape(-1, 1)
    msg = 'The covariance matrix of the support data is equal to 0, try to increase support_fraction'
    for X in [X_1, X_2]:
        with pytest.raises(ValueError, match=msg):
            MinCovDet().fit(X)