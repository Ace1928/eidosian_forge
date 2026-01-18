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
@pytest.mark.parametrize('C', [0.001, 1, 100, 1000000.0])
def test_elastic_net_vs_l1_l2(C):
    X, y = make_classification(500, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    param_grid = {'l1_ratio': np.linspace(0, 1, 5)}
    enet_clf = LogisticRegression(penalty='elasticnet', C=C, solver='saga', random_state=0, tol=0.01)
    gs = GridSearchCV(enet_clf, param_grid, refit=True)
    l1_clf = LogisticRegression(penalty='l1', C=C, solver='saga', random_state=0, tol=0.01)
    l2_clf = LogisticRegression(penalty='l2', C=C, solver='saga', random_state=0, tol=0.01)
    for clf in (gs, l1_clf, l2_clf):
        clf.fit(X_train, y_train)
    assert gs.score(X_test, y_test) >= l1_clf.score(X_test, y_test)
    assert gs.score(X_test, y_test) >= l2_clf.score(X_test, y_test)