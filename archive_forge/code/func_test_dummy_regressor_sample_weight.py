import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS
from sklearn.utils.stats import _weighted_percentile
def test_dummy_regressor_sample_weight(global_random_seed, n_samples=10):
    random_state = np.random.RandomState(seed=global_random_seed)
    X = [[0]] * n_samples
    y = random_state.rand(n_samples)
    sample_weight = random_state.rand(n_samples)
    est = DummyRegressor(strategy='mean').fit(X, y, sample_weight)
    assert est.constant_ == np.average(y, weights=sample_weight)
    est = DummyRegressor(strategy='median').fit(X, y, sample_weight)
    assert est.constant_ == _weighted_percentile(y, sample_weight, 50.0)
    est = DummyRegressor(strategy='quantile', quantile=0.95).fit(X, y, sample_weight)
    assert est.constant_ == _weighted_percentile(y, sample_weight, 95.0)