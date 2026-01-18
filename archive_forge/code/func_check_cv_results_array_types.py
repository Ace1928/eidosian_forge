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
def check_cv_results_array_types(search, param_keys, score_keys):
    cv_results = search.cv_results_
    assert all((isinstance(cv_results[param], np.ma.MaskedArray) for param in param_keys))
    assert all((cv_results[key].dtype == object for key in param_keys))
    assert not any((isinstance(cv_results[key], np.ma.MaskedArray) for key in score_keys))
    assert all((cv_results[key].dtype == np.float64 for key in score_keys if not key.startswith('rank')))
    scorer_keys = search.scorer_.keys() if search.multimetric_ else ['score']
    for key in scorer_keys:
        assert cv_results['rank_test_%s' % key].dtype == np.int32