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
def test_multimetric_scorer_sanity_check():
    scorers = {'a1': 'accuracy', 'a2': 'accuracy', 'll1': 'neg_log_loss', 'll2': 'neg_log_loss', 'ra1': 'roc_auc', 'ra2': 'roc_auc'}
    X, y = make_classification(random_state=0)
    clf = DecisionTreeClassifier()
    clf.fit(X, y)
    scorer_dict = _check_multimetric_scoring(clf, scorers)
    multi_scorer = _MultimetricScorer(scorers=scorer_dict)
    result = multi_scorer(clf, X, y)
    separate_scores = {name: get_scorer(name)(clf, X, y) for name in ['accuracy', 'neg_log_loss', 'roc_auc']}
    for key, value in result.items():
        score_name = scorers[key]
        assert_allclose(value, separate_scores[score_name])