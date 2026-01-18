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
@pytest.mark.parametrize('score_func', [f1_score, precision_score, recall_score, jaccard_score])
def test_non_symmetric_metric_pos_label(score_func, string_labeled_classification_problem):
    clf, X_test, y_test, y_pred, _, _ = string_labeled_classification_problem
    pos_label = 'cancer'
    assert clf.classes_[0] == pos_label
    score_pos_cancer = score_func(y_test, y_pred, pos_label='cancer')
    score_pos_not_cancer = score_func(y_test, y_pred, pos_label='not cancer')
    assert score_pos_cancer != pytest.approx(score_pos_not_cancer)
    scorer = make_scorer(score_func, pos_label=pos_label)
    assert scorer(clf, X_test, y_test) == pytest.approx(score_pos_cancer)