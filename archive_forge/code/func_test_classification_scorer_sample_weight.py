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
@ignore_warnings
def test_classification_scorer_sample_weight():
    X, y = make_classification(random_state=0)
    _, y_ml = make_multilabel_classification(n_samples=X.shape[0], random_state=0)
    split = train_test_split(X, y, y_ml, random_state=0)
    X_train, X_test, y_train, y_test, y_ml_train, y_ml_test = split
    sample_weight = np.ones_like(y_test)
    sample_weight[:10] = 0
    estimator = _make_estimators(X_train, y_train, y_ml_train)
    for name in get_scorer_names():
        scorer = get_scorer(name)
        if name in REGRESSION_SCORERS:
            continue
        if name == 'top_k_accuracy':
            scorer._kwargs = {'k': 1}
        if name in MULTILABEL_ONLY_SCORERS:
            target = y_ml_test
        else:
            target = y_test
        try:
            weighted = scorer(estimator[name], X_test, target, sample_weight=sample_weight)
            ignored = scorer(estimator[name], X_test[10:], target[10:])
            unweighted = scorer(estimator[name], X_test, target)
            _ = scorer(estimator[name], X_test[:10], target[:10], sample_weight=None)
            assert weighted != unweighted, f'scorer {name} behaves identically when called with sample weights: {weighted} vs {unweighted}'
            assert_almost_equal(weighted, ignored, err_msg=f'scorer {name} behaves differently when ignoring samples and setting sample_weight to 0: {weighted} vs {ignored}')
        except TypeError as e:
            assert 'sample_weight' in str(e), f'scorer {name} raises unhelpful exception when called with sample weights: {str(e)}'