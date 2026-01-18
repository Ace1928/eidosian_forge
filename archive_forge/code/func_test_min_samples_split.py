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
def test_min_samples_split(name):
    X, y = (hastie_X, hastie_y)
    ForestEstimator = FOREST_ESTIMATORS[name]
    est = ForestEstimator(min_samples_split=10, n_estimators=1, random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]
    assert np.min(node_samples) > len(X) * 0.5 - 1, 'Failed with {0}'.format(name)
    est = ForestEstimator(min_samples_split=0.5, n_estimators=1, random_state=0)
    est.fit(X, y)
    node_idx = est.estimators_[0].tree_.children_left != -1
    node_samples = est.estimators_[0].tree_.n_node_samples[node_idx]
    assert np.min(node_samples) > len(X) * 0.5 - 1, 'Failed with {0}'.format(name)