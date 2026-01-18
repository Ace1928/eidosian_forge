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
def test_iforest_parallel_regression(global_random_seed):
    """Check parallel regression."""
    rng = check_random_state(global_random_seed)
    X_train, X_test = train_test_split(diabetes.data, random_state=rng)
    ensemble = IsolationForest(n_jobs=3, random_state=global_random_seed).fit(X_train)
    ensemble.set_params(n_jobs=1)
    y1 = ensemble.predict(X_test)
    ensemble.set_params(n_jobs=2)
    y2 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y2)
    ensemble = IsolationForest(n_jobs=1, random_state=global_random_seed).fit(X_train)
    y3 = ensemble.predict(X_test)
    assert_array_almost_equal(y1, y3)