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
def test_logistic_regression_class_weights():
    X_iris = scale(iris.data)
    X = X_iris[45:, :]
    y = iris.target[45:]
    solvers = ('lbfgs', 'newton-cg')
    class_weight_dict = _compute_class_weight_dictionary(y)
    for solver in solvers:
        clf1 = LogisticRegression(solver=solver, multi_class='multinomial', class_weight='balanced')
        clf2 = LogisticRegression(solver=solver, multi_class='multinomial', class_weight=class_weight_dict)
        clf1.fit(X, y)
        clf2.fit(X, y)
        assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=4)
    X = X_iris[45:100, :]
    y = iris.target[45:100]
    class_weight_dict = _compute_class_weight_dictionary(y)
    for solver in set(SOLVERS) - set(('sag', 'saga')):
        clf1 = LogisticRegression(solver=solver, multi_class='ovr', class_weight='balanced')
        clf2 = LogisticRegression(solver=solver, multi_class='ovr', class_weight=class_weight_dict)
        clf1.fit(X, y)
        clf2.fit(X, y)
        assert_array_almost_equal(clf1.coef_, clf2.coef_, decimal=6)