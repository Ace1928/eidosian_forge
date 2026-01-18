import itertools
import os
import warnings
from functools import partial
import numpy as np
import pytest
from numpy.testing import (
from scipy import sparse
from sklearn import config_context
from sklearn.base import clone
from sklearn.datasets import load_iris, make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.linear_model._logistic import (
from sklearn.metrics import get_scorer, log_loss
from sklearn.model_selection import (
from sklearn.preprocessing import LabelEncoder, StandardScaler, scale
from sklearn.svm import l1_min_c
from sklearn.utils import _IS_32BIT, compute_class_weight, shuffle
from sklearn.utils._testing import ignore_warnings, skip_if_no_parallel
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
def test_multinomial_binary_probabilities(global_random_seed):
    X, y = make_classification(random_state=global_random_seed)
    clf = LogisticRegression(multi_class='multinomial', solver='saga', tol=0.001, random_state=global_random_seed)
    clf.fit(X, y)
    decision = clf.decision_function(X)
    proba = clf.predict_proba(X)
    expected_proba_class_1 = np.exp(decision) / (np.exp(decision) + np.exp(-decision))
    expected_proba = np.c_[1 - expected_proba_class_1, expected_proba_class_1]
    assert_almost_equal(proba, expected_proba)