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
def test_get_scorer_multilabel_indicator():
    """Check that our scorer deal with multi-label indicator matrices.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/26817
    """
    X, Y = make_multilabel_classification(n_samples=72, n_classes=3, random_state=0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
    estimator = KNeighborsClassifier().fit(X_train, Y_train)
    score = get_scorer('average_precision')(estimator, X_test, Y_test)
    assert score > 0.8