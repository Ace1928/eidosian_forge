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
def test_brier_score_loss_pos_label(string_labeled_classification_problem):
    clf, X_test, y_test, _, y_pred_proba, _ = string_labeled_classification_problem
    pos_label = 'cancer'
    assert clf.classes_[0] == pos_label
    brier_pos_cancer = brier_score_loss(y_test, y_pred_proba[:, 0], pos_label='cancer')
    brier_pos_not_cancer = brier_score_loss(y_test, y_pred_proba[:, 1], pos_label='not cancer')
    assert brier_pos_cancer == pytest.approx(brier_pos_not_cancer)
    brier_scorer = make_scorer(brier_score_loss, response_method='predict_proba', pos_label=pos_label)
    assert brier_scorer(clf, X_test, y_test) == pytest.approx(brier_pos_cancer)