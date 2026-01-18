import numbers
import os
import pickle
import shutil
import tempfile
from copy import deepcopy
from functools import partial
from unittest.mock import Mock
import joblib
import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.linear_model import LogisticRegression, Perceptron, Ridge
from sklearn.metrics import (
from sklearn.metrics import cluster as cluster_module
from sklearn.metrics._scorer import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._testing import (
from sklearn.utils.metadata_routing import MetadataRouter
@pytest.fixture
def string_labeled_classification_problem():
    """Train a classifier on binary problem with string target.

    The classifier is trained on a binary classification problem where the
    minority class of interest has a string label that is intentionally not the
    greatest class label using the lexicographic order. In this case, "cancer"
    is the positive label, and `classifier.classes_` is
    `["cancer", "not cancer"]`.

    In addition, the dataset is imbalanced to better identify problems when
    using non-symmetric performance metrics such as f1-score, average precision
    and so on.

    Returns
    -------
    classifier : estimator object
        Trained classifier on the binary problem.
    X_test : ndarray of shape (n_samples, n_features)
        Data to be used as testing set in tests.
    y_test : ndarray of shape (n_samples,), dtype=object
        Binary target where labels are strings.
    y_pred : ndarray of shape (n_samples,), dtype=object
        Prediction of `classifier` when predicting for `X_test`.
    y_pred_proba : ndarray of shape (n_samples, 2), dtype=np.float64
        Probabilities of `classifier` when predicting for `X_test`.
    y_pred_decision : ndarray of shape (n_samples,), dtype=np.float64
        Decision function values of `classifier` when predicting on `X_test`.
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.utils import shuffle
    X, y = load_breast_cancer(return_X_y=True)
    idx_positive = np.flatnonzero(y == 1)
    idx_negative = np.flatnonzero(y == 0)
    idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    X, y = (X[idx_selected], y[idx_selected])
    X, y = shuffle(X, y, random_state=42)
    X = X[:, :2]
    y = np.array(['cancer' if c == 1 else 'not cancer' for c in y], dtype=object)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
    classifier = LogisticRegression().fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    y_pred_decision = classifier.decision_function(X_test)
    return (classifier, X_test, y_test, y_pred, y_pred_proba, y_pred_decision)