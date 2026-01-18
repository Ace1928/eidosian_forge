import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.parametrize('use_sparse', [False, True])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_cross_validate(use_sparse: bool, csr_container):
    cv = KFold()
    X_reg, y_reg = make_regression(n_samples=30, random_state=0)
    reg = Ridge(random_state=0)
    X_clf, y_clf = make_classification(n_samples=30, random_state=0)
    clf = SVC(kernel='linear', random_state=0)
    if use_sparse:
        X_reg = csr_container(X_reg)
        X_clf = csr_container(X_clf)
    for X, y, est in ((X_reg, y_reg, reg), (X_clf, y_clf, clf)):
        mse_scorer = check_scoring(est, scoring='neg_mean_squared_error')
        r2_scorer = check_scoring(est, scoring='r2')
        train_mse_scores = []
        test_mse_scores = []
        train_r2_scores = []
        test_r2_scores = []
        fitted_estimators = []
        for train, test in cv.split(X, y):
            est = clone(est).fit(X[train], y[train])
            train_mse_scores.append(mse_scorer(est, X[train], y[train]))
            train_r2_scores.append(r2_scorer(est, X[train], y[train]))
            test_mse_scores.append(mse_scorer(est, X[test], y[test]))
            test_r2_scores.append(r2_scorer(est, X[test], y[test]))
            fitted_estimators.append(est)
        train_mse_scores = np.array(train_mse_scores)
        test_mse_scores = np.array(test_mse_scores)
        train_r2_scores = np.array(train_r2_scores)
        test_r2_scores = np.array(test_r2_scores)
        fitted_estimators = np.array(fitted_estimators)
        scores = (train_mse_scores, test_mse_scores, train_r2_scores, test_r2_scores, fitted_estimators)
        check_cross_validate_single_metric(est, X, y, scores, cv)
        check_cross_validate_multi_metric(est, X, y, scores, cv)