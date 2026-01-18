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
@pytest.mark.parametrize('name', FOREST_ESTIMATORS)
def test_min_weight_fraction_leaf(name):
    X, y = (hastie_X, hastie_y)
    ForestEstimator = FOREST_ESTIMATORS[name]
    rng = np.random.RandomState(0)
    weights = rng.rand(X.shape[0])
    total_weight = np.sum(weights)
    for frac in np.linspace(0, 0.5, 6):
        est = ForestEstimator(min_weight_fraction_leaf=frac, n_estimators=1, random_state=0)
        if 'RandomForest' in name:
            est.bootstrap = False
        est.fit(X, y, sample_weight=weights)
        out = est.estimators_[0].tree_.apply(X)
        node_weights = np.bincount(out, weights=weights)
        leaf_weights = node_weights[node_weights != 0]
        assert np.min(leaf_weights) >= total_weight * est.min_weight_fraction_leaf, 'Failed with {0} min_weight_fraction_leaf={1}'.format(name, est.min_weight_fraction_leaf)