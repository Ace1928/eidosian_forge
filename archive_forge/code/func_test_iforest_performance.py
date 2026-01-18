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
def test_iforest_performance(global_random_seed):
    """Test Isolation Forest performs well"""
    rng = check_random_state(global_random_seed)
    X = 0.3 * rng.randn(600, 2)
    X = rng.permutation(np.vstack((X + 2, X - 2)))
    X_train = X[:1000]
    X_outliers = rng.uniform(low=-1, high=1, size=(200, 2))
    X_test = np.vstack((X[1000:], X_outliers))
    y_test = np.array([0] * 200 + [1] * 200)
    clf = IsolationForest(max_samples=100, random_state=rng).fit(X_train)
    y_pred = -clf.decision_function(X_test)
    assert roc_auc_score(y_test, y_pred) > 0.98