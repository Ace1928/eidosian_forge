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
def test_callable_multimetric_error_failing_clf():

    def custom_scorer(est, X, y):
        return {'acc': 1}
    X, y = make_classification(n_samples=20, n_features=10, random_state=0)
    clf = FailingClassifier()
    gs = GridSearchCV(clf, [{'parameter': [0, 1, 2]}], scoring=custom_scorer, refit=False, error_score=0.1)
    warning_message = re.compile('5 fits failed.+total of 15.+The score on these train-test partitions for these parameters will be set to 0\\.1', flags=re.DOTALL)
    with pytest.warns(FitFailedWarning, match=warning_message):
        gs.fit(X, y)
    assert_allclose(gs.cv_results_['mean_test_acc'], [1, 1, 0.1])