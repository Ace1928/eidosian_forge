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
@ignore_warnings
def test_grid_search_no_score():
    clf = LinearSVC(dual='auto', random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    clf_no_score = LinearSVCNoScore(dual='auto', random_state=0)
    grid_search = GridSearchCV(clf, {'C': Cs}, scoring='accuracy')
    grid_search.fit(X, y)
    grid_search_no_score = GridSearchCV(clf_no_score, {'C': Cs}, scoring='accuracy')
    grid_search_no_score.fit(X, y)
    assert grid_search_no_score.best_params_ == grid_search.best_params_
    assert grid_search.score(X, y) == grid_search_no_score.score(X, y)
    grid_search_no_score = GridSearchCV(clf_no_score, {'C': Cs})
    with pytest.raises(TypeError, match='no scoring'):
        grid_search_no_score.fit([[1]])