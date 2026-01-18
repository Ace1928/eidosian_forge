import pickle
import re
import sys
from collections.abc import Iterable, Sized
from functools import partial
from io import StringIO
from itertools import chain, product
from types import GeneratorType
import numpy as np
import pytest
from scipy.stats import bernoulli, expon, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.experimental import enable_halving_search_cv  # noqa
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import (
from sklearn.model_selection._search import BaseSearchCV
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.neighbors import KernelDensity, KNeighborsClassifier, LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
def test_grid_search_score_method():
    X, y = make_classification(n_samples=100, n_classes=2, flip_y=0.2, random_state=0)
    clf = LinearSVC(dual='auto', random_state=0)
    grid = {'C': [0.1]}
    search_no_scoring = GridSearchCV(clf, grid, scoring=None).fit(X, y)
    search_accuracy = GridSearchCV(clf, grid, scoring='accuracy').fit(X, y)
    search_no_score_method_auc = GridSearchCV(LinearSVCNoScore(dual='auto'), grid, scoring='roc_auc').fit(X, y)
    search_auc = GridSearchCV(clf, grid, scoring='roc_auc').fit(X, y)
    score_no_scoring = search_no_scoring.score(X, y)
    score_accuracy = search_accuracy.score(X, y)
    score_no_score_auc = search_no_score_method_auc.score(X, y)
    score_auc = search_auc.score(X, y)
    assert score_auc < 1.0
    assert score_accuracy < 1.0
    assert score_auc != score_accuracy
    assert_almost_equal(score_accuracy, score_no_scoring)
    assert_almost_equal(score_auc, score_no_score_auc)