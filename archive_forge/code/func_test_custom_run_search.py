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
def test_custom_run_search():

    def check_results(results, gscv):
        exp_results = gscv.cv_results_
        assert sorted(results.keys()) == sorted(exp_results)
        for k in results:
            if not k.endswith('_time'):
                results[k] = np.asanyarray(results[k])
                if results[k].dtype.kind == 'O':
                    assert_array_equal(exp_results[k], results[k], err_msg='Checking ' + k)
                else:
                    assert_allclose(exp_results[k], results[k], err_msg='Checking ' + k)

    def fit_grid(param_grid):
        return GridSearchCV(clf, param_grid, return_train_score=True).fit(X, y)

    class CustomSearchCV(BaseSearchCV):

        def __init__(self, estimator, **kwargs):
            super().__init__(estimator, **kwargs)

        def _run_search(self, evaluate):
            results = evaluate([{'max_depth': 1}, {'max_depth': 2}])
            check_results(results, fit_grid({'max_depth': [1, 2]}))
            results = evaluate([{'min_samples_split': 5}, {'min_samples_split': 10}])
            check_results(results, fit_grid([{'max_depth': [1, 2]}, {'min_samples_split': [5, 10]}]))
    clf = DecisionTreeRegressor(random_state=0)
    X, y = make_classification(n_samples=100, n_informative=4, random_state=0)
    mycv = CustomSearchCV(clf, return_train_score=True).fit(X, y)
    gscv = fit_grid([{'max_depth': [1, 2]}, {'min_samples_split': [5, 10]}])
    results = mycv.cv_results_
    check_results(results, gscv)
    for attr in dir(gscv):
        if attr[0].islower() and attr[-1:] == '_' and (attr not in {'cv_results_', 'best_estimator_', 'refit_time_', 'classes_', 'scorer_'}):
            assert getattr(gscv, attr) == getattr(mycv, attr), 'Attribute %s not equal' % attr