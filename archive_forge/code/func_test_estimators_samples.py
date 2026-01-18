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
@pytest.mark.parametrize('seed', [None, 1])
@pytest.mark.parametrize('bootstrap', [True, False])
@pytest.mark.parametrize('ForestClass', FOREST_CLASSIFIERS_REGRESSORS.values())
def test_estimators_samples(ForestClass, bootstrap, seed):
    """Estimators_samples_ property should be consistent.

    Tests consistency across fits and whether or not the seed for the random generator
    is set.
    """
    X, y = make_hastie_10_2(n_samples=200, random_state=1)
    if bootstrap:
        max_samples = 0.5
    else:
        max_samples = None
    est = ForestClass(n_estimators=10, max_samples=max_samples, max_features=0.5, random_state=seed, bootstrap=bootstrap)
    est.fit(X, y)
    estimators_samples = est.estimators_samples_.copy()
    assert_array_equal(estimators_samples, est.estimators_samples_)
    estimators = est.estimators_
    assert isinstance(estimators_samples, list)
    assert len(estimators_samples) == len(estimators)
    assert estimators_samples[0].dtype == np.int32
    for i in range(len(estimators)):
        if bootstrap:
            assert len(estimators_samples[i]) == len(X) // 2
            assert len(np.unique(estimators_samples[i])) < len(estimators_samples[i])
        else:
            assert len(set(estimators_samples[i])) == len(X)
    estimator_index = 0
    estimator_samples = estimators_samples[estimator_index]
    estimator = estimators[estimator_index]
    X_train = X[estimator_samples]
    y_train = y[estimator_samples]
    orig_tree_values = estimator.tree_.value
    estimator = clone(estimator)
    estimator.fit(X_train, y_train)
    new_tree_values = estimator.tree_.value
    assert_allclose(orig_tree_values, new_tree_values)