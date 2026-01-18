import warnings
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn import clone
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils._testing import (
def test_invalid_n_bins_array():
    n_bins = np.full((2, 4), 2.0)
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = 'n_bins must be a scalar or array of shape \\(n_features,\\).'
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)
    n_bins = [1, 2, 2]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = 'n_bins must be a scalar or array of shape \\(n_features,\\).'
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)
    n_bins = [1, 2, 2, 1]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = 'KBinsDiscretizer received an invalid number of bins at indices 0, 3. Number of bins must be at least 2, and must be an int.'
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)
    n_bins = [2.1, 2, 2.1, 2]
    est = KBinsDiscretizer(n_bins=n_bins)
    err_msg = 'KBinsDiscretizer received an invalid number of bins at indices 0, 2. Number of bins must be at least 2, and must be an int.'
    with pytest.raises(ValueError, match=err_msg):
        est.fit_transform(X)