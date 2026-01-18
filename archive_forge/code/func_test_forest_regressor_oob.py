import itertools
import math
import pickle
from collections import defaultdict
from functools import partial
from itertools import combinations, product
from typing import Any, Dict
from unittest.mock import patch
import joblib
import numpy as np
import pytest
from scipy.special import comb
import sklearn
from sklearn import clone, datasets
from sklearn.datasets import make_classification, make_hastie_10_2
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.ensemble._forest import (
from sklearn.exceptions import NotFittedError
from sklearn.metrics import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.svm import LinearSVC
from sklearn.tree._classes import SPARSE_SPLITTERS
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.multiclass import type_of_target
from sklearn.utils.parallel import Parallel
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('ForestRegressor', FOREST_REGRESSORS.values())
@pytest.mark.parametrize('X_type', ['array', 'sparse_csr', 'sparse_csc'])
@pytest.mark.parametrize('X, y, lower_bound_r2', [(*datasets.make_regression(n_samples=500, n_features=10, n_targets=1, random_state=0), 0.7), (*datasets.make_regression(n_samples=500, n_features=10, n_targets=2, random_state=0), 0.55)])
@pytest.mark.parametrize('oob_score', [True, explained_variance_score])
def test_forest_regressor_oob(ForestRegressor, X, y, X_type, lower_bound_r2, oob_score):
    """Check that forest-based regressor provide an OOB score close to the
    score on a test set."""
    X = _convert_container(X, constructor_name=X_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
    regressor = ForestRegressor(n_estimators=50, bootstrap=True, oob_score=oob_score, random_state=0)
    assert not hasattr(regressor, 'oob_score_')
    assert not hasattr(regressor, 'oob_prediction_')
    regressor.fit(X_train, y_train)
    if callable(oob_score):
        test_score = oob_score(y_test, regressor.predict(X_test))
    else:
        test_score = regressor.score(X_test, y_test)
        assert regressor.oob_score_ >= lower_bound_r2
    assert abs(test_score - regressor.oob_score_) <= 0.1
    assert hasattr(regressor, 'oob_score_')
    assert hasattr(regressor, 'oob_prediction_')
    assert not hasattr(regressor, 'oob_decision_function_')
    if y.ndim == 1:
        expected_shape = (X_train.shape[0],)
    else:
        expected_shape = (X_train.shape[0], y.ndim)
    assert regressor.oob_prediction_.shape == expected_shape