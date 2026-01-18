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
def test_multimetric_scorer_calls_method_once_regressor_threshold():
    predict_called_cnt = 0

    class MockDecisionTreeRegressor(DecisionTreeRegressor):

        def predict(self, X):
            nonlocal predict_called_cnt
            predict_called_cnt += 1
            return super().predict(X)
    X, y = (np.array([[1], [1], [0], [0], [0]]), np.array([0, 1, 1, 1, 0]))
    clf = MockDecisionTreeRegressor()
    clf.fit(X, y)
    scorers = {'neg_mse': 'neg_mean_squared_error', 'r2': 'r2'}
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    scorer = _MultimetricScorer(scorers=scorer_dict)
    scorer(clf, X, y)
    assert predict_called_cnt == 1