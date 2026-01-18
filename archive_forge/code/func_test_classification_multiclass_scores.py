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
@pytest.mark.parametrize('scorer_name, metric', [('accuracy', accuracy_score), ('balanced_accuracy', balanced_accuracy_score), ('f1_weighted', partial(f1_score, average='weighted')), ('f1_macro', partial(f1_score, average='macro')), ('f1_micro', partial(f1_score, average='micro')), ('precision_weighted', partial(precision_score, average='weighted')), ('precision_macro', partial(precision_score, average='macro')), ('precision_micro', partial(precision_score, average='micro')), ('recall_weighted', partial(recall_score, average='weighted')), ('recall_macro', partial(recall_score, average='macro')), ('recall_micro', partial(recall_score, average='micro')), ('jaccard_weighted', partial(jaccard_score, average='weighted')), ('jaccard_macro', partial(jaccard_score, average='macro')), ('jaccard_micro', partial(jaccard_score, average='micro'))])
def test_classification_multiclass_scores(scorer_name, metric):
    X, y = make_classification(n_classes=3, n_informative=3, n_samples=30, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    score = get_scorer(scorer_name)(clf, X_test, y_test)
    expected_score = metric(y_test, clf.predict(X_test))
    assert score == pytest.approx(expected_score)