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
def test_grid_search_correct_score_results():
    n_splits = 3
    clf = LinearSVC(dual='auto', random_state=0)
    X, y = make_blobs(random_state=0, centers=2)
    Cs = [0.1, 1, 10]
    for score in ['f1', 'roc_auc']:
        grid_search = GridSearchCV(clf, {'C': Cs}, scoring=score, cv=n_splits)
        cv_results = grid_search.fit(X, y).cv_results_
        result_keys = list(cv_results.keys())
        expected_keys = ('mean_test_score', 'rank_test_score') + tuple(('split%d_test_score' % cv_i for cv_i in range(n_splits)))
        assert all(np.isin(expected_keys, result_keys))
        cv = StratifiedKFold(n_splits=n_splits)
        n_splits = grid_search.n_splits_
        for candidate_i, C in enumerate(Cs):
            clf.set_params(C=C)
            cv_scores = np.array([grid_search.cv_results_['split%d_test_score' % s][candidate_i] for s in range(n_splits)])
            for i, (train, test) in enumerate(cv.split(X, y)):
                clf.fit(X[train], y[train])
                if score == 'f1':
                    correct_score = f1_score(y[test], clf.predict(X[test]))
                elif score == 'roc_auc':
                    dec = clf.decision_function(X[test])
                    correct_score = roc_auc_score(y[test], dec)
                assert_almost_equal(correct_score, cv_scores[i])