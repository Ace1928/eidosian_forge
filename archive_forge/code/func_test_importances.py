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
@pytest.mark.parametrize('dtype', (np.float64, np.float32))
@pytest.mark.parametrize('name, criterion', itertools.chain(product(FOREST_CLASSIFIERS, ['gini', 'log_loss']), product(FOREST_REGRESSORS, ['squared_error', 'friedman_mse', 'absolute_error'])))
def test_importances(dtype, name, criterion):
    tolerance = 0.01
    if name in FOREST_REGRESSORS and criterion == 'absolute_error':
        tolerance = 0.05
    X = X_large.astype(dtype, copy=False)
    y = y_large.astype(dtype, copy=False)
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=10, criterion=criterion, random_state=0)
    est.fit(X, y)
    importances = est.feature_importances_
    n_important = np.sum(importances > 0.1)
    assert importances.shape[0] == 10
    assert n_important == 3
    assert np.all(importances[:3] > 0.1)
    importances = est.feature_importances_
    est.set_params(n_jobs=2)
    importances_parallel = est.feature_importances_
    assert_array_almost_equal(importances, importances_parallel)
    sample_weight = check_random_state(0).randint(1, 10, len(X))
    est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
    est.fit(X, y, sample_weight=sample_weight)
    importances = est.feature_importances_
    assert np.all(importances >= 0.0)
    for scale in [0.5, 100]:
        est = ForestEstimator(n_estimators=10, random_state=0, criterion=criterion)
        est.fit(X, y, sample_weight=scale * sample_weight)
        importances_bis = est.feature_importances_
        assert np.abs(importances - importances_bis).mean() < tolerance