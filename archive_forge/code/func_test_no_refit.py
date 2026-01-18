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
def test_no_refit():
    clf = MockClassifier()
    for scoring in [None, ['accuracy', 'precision']]:
        grid_search = GridSearchCV(clf, {'foo_param': [1, 2, 3]}, refit=False, cv=3)
        grid_search.fit(X, y)
        assert not hasattr(grid_search, 'best_estimator_') and hasattr(grid_search, 'best_index_') and hasattr(grid_search, 'best_params_')
        for fn_name in ('predict', 'predict_proba', 'predict_log_proba', 'transform', 'inverse_transform'):
            outer_msg = f"has no attribute '{fn_name}'"
            inner_msg = f'`refit=False`. {fn_name} is available only after refitting on the best parameters'
            with pytest.raises(AttributeError, match=outer_msg) as exec_info:
                getattr(grid_search, fn_name)(X)
            assert isinstance(exec_info.value.__cause__, AttributeError)
            assert inner_msg in str(exec_info.value.__cause__)
    error_msg = 'For multi-metric scoring, the parameter refit must be set to a scorer key'
    for refit in [True, 'recall', 'accuracy']:
        with pytest.raises(ValueError, match=error_msg):
            GridSearchCV(clf, {}, refit=refit, scoring={'acc': 'accuracy', 'prec': 'precision'}).fit(X, y)