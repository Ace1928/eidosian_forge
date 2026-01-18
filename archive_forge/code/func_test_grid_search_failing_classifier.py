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
def test_grid_search_failing_classifier():
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)
    clf = FailingClassifier()
    gs = GridSearchCV(clf, [{'parameter': [0, 1, 2]}], scoring='accuracy', refit=False, error_score=0.0)
    warning_message = re.compile('5 fits failed.+total of 15.+The score on these train-test partitions for these parameters will be set to 0\\.0.+5 fits failed with the following error.+ValueError.+Failing classifier failed as required', flags=re.DOTALL)
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)
    n_candidates = len(gs.cv_results_['params'])

    def get_cand_scores(i):
        return np.array([gs.cv_results_['split%d_test_score' % s][i] for s in range(gs.n_splits_)])
    assert all((np.all(get_cand_scores(cand_i) == 0.0) for cand_i in range(n_candidates) if gs.cv_results_['param_parameter'][cand_i] == FailingClassifier.FAILING_PARAMETER))
    gs = GridSearchCV(clf, [{'parameter': [0, 1, 2]}], scoring='accuracy', refit=False, error_score=float('nan'))
    warning_message = re.compile('5 fits failed.+total of 15.+The score on these train-test partitions for these parameters will be set to nan.+5 fits failed with the following error.+ValueError.+Failing classifier failed as required', flags=re.DOTALL)
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)
    n_candidates = len(gs.cv_results_['params'])
    assert all((np.all(np.isnan(get_cand_scores(cand_i))) for cand_i in range(n_candidates) if gs.cv_results_['param_parameter'][cand_i] == FailingClassifier.FAILING_PARAMETER))
    ranks = gs.cv_results_['rank_test_score']
    assert ranks[0] <= 2 and ranks[1] <= 2
    assert ranks[clf.FAILING_PARAMETER] == 3
    assert gs.best_index_ != clf.FAILING_PARAMETER