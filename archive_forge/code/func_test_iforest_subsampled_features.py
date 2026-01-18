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
def test_iforest_subsampled_features():
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(diabetes.data[:50], diabetes.target[:50], random_state=rng)
    clf = IsolationForest(max_features=0.8)
    clf.fit(X_train, y_train)
    clf.predict(X_test)