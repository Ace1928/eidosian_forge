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
def test_LogisticRegressionCV_elasticnet_attribute_shapes():
    n_classes = 3
    n_features = 20
    X, y = make_classification(n_samples=200, n_classes=n_classes, n_informative=n_classes, n_features=n_features, random_state=0)
    Cs = np.logspace(-4, 4, 3)
    l1_ratios = np.linspace(0, 1, 2)
    n_folds = 2
    lrcv = LogisticRegressionCV(penalty='elasticnet', Cs=Cs, solver='saga', cv=n_folds, l1_ratios=l1_ratios, multi_class='ovr', random_state=0, tol=0.01)
    lrcv.fit(X, y)
    coefs_paths = np.asarray(list(lrcv.coefs_paths_.values()))
    assert coefs_paths.shape == (n_classes, n_folds, Cs.size, l1_ratios.size, n_features + 1)
    scores = np.asarray(list(lrcv.scores_.values()))
    assert scores.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)
    assert lrcv.n_iter_.shape == (n_classes, n_folds, Cs.size, l1_ratios.size)