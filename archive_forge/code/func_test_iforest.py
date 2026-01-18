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
def test_iforest(global_random_seed):
    """Check Isolation Forest for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])
    grid = ParameterGrid({'n_estimators': [3], 'max_samples': [0.5, 1.0, 3], 'bootstrap': [True, False]})
    with ignore_warnings():
        for params in grid:
            IsolationForest(random_state=global_random_seed, **params).fit(X_train).predict(X_test)