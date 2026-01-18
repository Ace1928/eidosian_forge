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
def test_max_samples_attribute():
    X = iris.data
    clf = IsolationForest().fit(X)
    assert clf.max_samples_ == X.shape[0]
    clf = IsolationForest(max_samples=500)
    warn_msg = 'max_samples will be set to n_samples for estimation'
    with pytest.warns(UserWarning, match=warn_msg):
        clf.fit(X)
    assert clf.max_samples_ == X.shape[0]
    clf = IsolationForest(max_samples=0.4).fit(X)
    assert clf.max_samples_ == 0.4 * X.shape[0]