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
@pytest.mark.parametrize('name', FOREST_REGRESSORS)
def test_max_samples_boundary_regressors(name):
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, train_size=0.7, test_size=0.3, random_state=0)
    ms_1_model = FOREST_REGRESSORS[name](bootstrap=True, max_samples=1.0, random_state=0)
    ms_1_predict = ms_1_model.fit(X_train, y_train).predict(X_test)
    ms_None_model = FOREST_REGRESSORS[name](bootstrap=True, max_samples=None, random_state=0)
    ms_None_predict = ms_None_model.fit(X_train, y_train).predict(X_test)
    ms_1_ms = mean_squared_error(ms_1_predict, y_test)
    ms_None_ms = mean_squared_error(ms_None_predict, y_test)
    assert ms_1_ms == pytest.approx(ms_None_ms)