import re
import warnings
from itertools import combinations, combinations_with_replacement, permutations
import numpy as np
import pytest
from scipy import stats
from scipy.sparse import issparse
from scipy.special import comb
from sklearn import config_context
from sklearn.datasets import load_digits, make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import (
from sklearn.model_selection._split import (
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import assert_request_is_empty
from sklearn.utils._array_api import (
from sklearn.utils._array_api import (
from sklearn.utils._mocking import MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_kfold_can_detect_dependent_samples_on_digits():
    X, y = (digits.data[:600], digits.target[:600])
    model = SVC(C=10, gamma=0.005)
    n_splits = 3
    cv = KFold(n_splits=n_splits, shuffle=False)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert 0.92 > mean_score
    assert mean_score > 0.8
    cv = KFold(n_splits, shuffle=True, random_state=0)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert mean_score > 0.92
    cv = KFold(n_splits, shuffle=True, random_state=1)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert mean_score > 0.92
    cv = StratifiedKFold(n_splits)
    mean_score = cross_val_score(model, X, y, cv=cv).mean()
    assert 0.94 > mean_score
    assert mean_score > 0.8