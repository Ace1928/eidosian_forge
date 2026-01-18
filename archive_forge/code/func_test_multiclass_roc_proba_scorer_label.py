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
def test_multiclass_roc_proba_scorer_label():
    scorer = make_scorer(roc_auc_score, multi_class='ovo', labels=[0, 1, 2], response_method='predict_proba')
    X, y = make_classification(n_classes=3, n_informative=3, n_samples=20, random_state=0)
    lr = LogisticRegression(multi_class='multinomial').fit(X, y)
    y_proba = lr.predict_proba(X)
    y_binary = y == 0
    expected_score = roc_auc_score(y_binary, y_proba, multi_class='ovo', labels=[0, 1, 2])
    assert scorer(lr, X, y_binary) == pytest.approx(expected_score)