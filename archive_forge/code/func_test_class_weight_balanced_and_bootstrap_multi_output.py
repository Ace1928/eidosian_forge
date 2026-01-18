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
def test_class_weight_balanced_and_bootstrap_multi_output(name):
    ForestClassifier = FOREST_CLASSIFIERS[name]
    _y = np.vstack((y, np.array(y) * 2)).T
    clf = ForestClassifier(class_weight='balanced', random_state=0)
    clf.fit(X, _y)
    clf = ForestClassifier(class_weight=[{-1: 0.5, 1: 1.0}, {-2: 1.0, 2: 1.0}], random_state=0)
    clf.fit(X, _y)
    clf = ForestClassifier(class_weight='balanced_subsample', random_state=0)
    clf.fit(X, _y)