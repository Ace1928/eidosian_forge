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
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS_REGRESSORS)
def test_decision_path(name):
    X, y = (hastie_X, hastie_y)
    n_samples = X.shape[0]
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(n_estimators=5, max_depth=1, warm_start=False, random_state=1)
    est.fit(X, y)
    indicator, n_nodes_ptr = est.decision_path(X)
    assert indicator.shape[1] == n_nodes_ptr[-1]
    assert indicator.shape[0] == n_samples
    assert_array_equal(np.diff(n_nodes_ptr), [e.tree_.node_count for e in est.estimators_])
    leaves = est.apply(X)
    for est_id in range(leaves.shape[1]):
        leave_indicator = [indicator[i, n_nodes_ptr[est_id] + j] for i, j in enumerate(leaves[:, est_id])]
        assert_array_almost_equal(leave_indicator, np.ones(shape=n_samples))