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
def test_grid_search_cv_splits_consistency():
    n_samples = 100
    n_splits = 5
    X, y = make_classification(n_samples=n_samples, random_state=0)
    gs = GridSearchCV(LinearSVC(dual='auto', random_state=0), param_grid={'C': [0.1, 0.2, 0.3]}, cv=OneTimeSplitter(n_splits=n_splits, n_samples=n_samples), return_train_score=True)
    gs.fit(X, y)
    gs2 = GridSearchCV(LinearSVC(dual='auto', random_state=0), param_grid={'C': [0.1, 0.2, 0.3]}, cv=KFold(n_splits=n_splits), return_train_score=True)
    gs2.fit(X, y)
    assert isinstance(KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y), GeneratorType)
    gs3 = GridSearchCV(LinearSVC(dual='auto', random_state=0), param_grid={'C': [0.1, 0.2, 0.3]}, cv=KFold(n_splits=n_splits, shuffle=True, random_state=0).split(X, y), return_train_score=True)
    gs3.fit(X, y)
    gs4 = GridSearchCV(LinearSVC(dual='auto', random_state=0), param_grid={'C': [0.1, 0.2, 0.3]}, cv=KFold(n_splits=n_splits, shuffle=True, random_state=0), return_train_score=True)
    gs4.fit(X, y)

    def _pop_time_keys(cv_results):
        for key in ('mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time'):
            cv_results.pop(key)
        return cv_results
    np.testing.assert_equal(_pop_time_keys(gs3.cv_results_), _pop_time_keys(gs4.cv_results_))
    np.testing.assert_equal({k: v for k, v in gs.cv_results_.items() if not k.endswith('_time')}, {k: v for k, v in gs2.cv_results_.items() if not k.endswith('_time')})
    gs = GridSearchCV(LinearSVC(dual='auto', random_state=0), param_grid={'C': [0.1, 0.1, 0.2, 0.2]}, cv=KFold(n_splits=n_splits, shuffle=True), return_train_score=True)
    gs.fit(X, y)
    for score_type in ('train', 'test'):
        per_param_scores = {}
        for param_i in range(4):
            per_param_scores[param_i] = [gs.cv_results_['split%d_%s_score' % (s, score_type)][param_i] for s in range(5)]
        assert_array_almost_equal(per_param_scores[0], per_param_scores[1])
        assert_array_almost_equal(per_param_scores[2], per_param_scores[3])