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
def test_kwargs_without_metadata_routing_error():

    def score(y_true, y_pred, param=None):
        return 1
    X, y = make_classification(n_samples=50, n_features=2, n_redundant=0, random_state=0)
    clf = DecisionTreeClassifier().fit(X, y)
    scorer = make_scorer(score)
    with config_context(enable_metadata_routing=False):
        with pytest.raises(ValueError, match='is only supported if enable_metadata_routing=True'):
            scorer(clf, X, y, param='blah')