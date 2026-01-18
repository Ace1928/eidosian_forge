import warnings
from unittest.mock import Mock, patch
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_iris, make_classification
from sklearn.ensemble import IsolationForest
from sklearn.ensemble._iforest import _average_path_length
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, train_test_split
from sklearn.utils import check_random_state
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('sparse_container', CSC_CONTAINERS + CSR_CONTAINERS)
def test_iforest_sparse(global_random_seed, sparse_container):
    """Check IForest for various parameter settings on sparse input."""
    rng = check_random_state(global_random_seed)
    X_train, X_test = train_test_split(diabetes.data[:50], random_state=rng)
    grid = ParameterGrid({'max_samples': [0.5, 1.0], 'bootstrap': [True, False]})
    X_train_sparse = sparse_container(X_train)
    X_test_sparse = sparse_container(X_test)
    for params in grid:
        sparse_classifier = IsolationForest(n_estimators=10, random_state=global_random_seed, **params).fit(X_train_sparse)
        sparse_results = sparse_classifier.predict(X_test_sparse)
        dense_classifier = IsolationForest(n_estimators=10, random_state=global_random_seed, **params).fit(X_train)
        dense_results = dense_classifier.predict(X_test)
        assert_array_equal(sparse_results, dense_results)