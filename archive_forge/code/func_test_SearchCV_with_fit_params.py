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
@pytest.mark.parametrize('SearchCV', [GridSearchCV, RandomizedSearchCV])
def test_SearchCV_with_fit_params(SearchCV):
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)
    clf = CheckingClassifier(expected_fit_params=['spam', 'eggs'])
    searcher = SearchCV(clf, {'foo_param': [1, 2, 3]}, cv=2, error_score='raise')
    err_msg = "Expected fit parameter\\(s\\) \\['eggs'\\] not seen."
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(10))
    err_msg = 'Fit parameter spam has length 1; expected'
    with pytest.raises(AssertionError, match=err_msg):
        searcher.fit(X, y, spam=np.ones(1), eggs=np.zeros(10))
    searcher.fit(X, y, spam=np.ones(10), eggs=np.zeros(10))