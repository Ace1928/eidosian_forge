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
@pytest.mark.parametrize('name', FOREST_CLASSIFIERS)
def test_multioutput_string(name):
    X_train = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1], [-2, 1], [-1, 1], [-1, 2], [2, -1], [1, -1], [1, -2]]
    y_train = [['red', 'blue'], ['red', 'blue'], ['red', 'blue'], ['green', 'green'], ['green', 'green'], ['green', 'green'], ['red', 'purple'], ['red', 'purple'], ['red', 'purple'], ['green', 'yellow'], ['green', 'yellow'], ['green', 'yellow']]
    X_test = [[-1, -1], [1, 1], [-1, 1], [1, -1]]
    y_test = [['red', 'blue'], ['green', 'green'], ['red', 'purple'], ['green', 'yellow']]
    est = FOREST_ESTIMATORS[name](random_state=0, bootstrap=False)
    y_pred = est.fit(X_train, y_train).predict(X_test)
    assert_array_equal(y_pred, y_test)
    with np.errstate(divide='ignore'):
        proba = est.predict_proba(X_test)
        assert len(proba) == 2
        assert proba[0].shape == (4, 2)
        assert proba[1].shape == (4, 4)
        log_proba = est.predict_log_proba(X_test)
        assert len(log_proba) == 2
        assert log_proba[0].shape == (4, 2)
        assert log_proba[1].shape == (4, 4)