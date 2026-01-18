import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_constant_size_multioutput_regressor(global_random_seed):
    random_state = np.random.RandomState(seed=global_random_seed)
    X = random_state.randn(10, 10)
    y = random_state.randn(10, 5)
    est = DummyRegressor(strategy='constant', constant=[1, 2, 3, 4])
    err_msg = 'Constant target value should have shape \\(5, 1\\).'
    with pytest.raises(ValueError, match=err_msg):
        est.fit(X, y)