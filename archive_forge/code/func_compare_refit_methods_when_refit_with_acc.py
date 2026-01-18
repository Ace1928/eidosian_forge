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
def compare_refit_methods_when_refit_with_acc(search_multi, search_acc, refit):
    """Compare refit multi-metric search methods with single metric methods"""
    assert search_acc.refit == refit
    if refit:
        assert search_multi.refit == 'accuracy'
    else:
        assert not search_multi.refit
        return
    X, y = make_blobs(n_samples=100, n_features=4, random_state=42)
    for method in ('predict', 'predict_proba', 'predict_log_proba'):
        assert_almost_equal(getattr(search_multi, method)(X), getattr(search_acc, method)(X))
    assert_almost_equal(search_multi.score(X, y), search_acc.score(X, y))
    for key in ('best_index_', 'best_score_', 'best_params_'):
        assert getattr(search_multi, key) == getattr(search_acc, key)