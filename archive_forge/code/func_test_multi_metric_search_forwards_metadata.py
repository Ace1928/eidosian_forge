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
@pytest.mark.usefixtures('enable_slep006')
@pytest.mark.parametrize('SearchCV, param_search', [(GridSearchCV, 'param_grid'), (RandomizedSearchCV, 'param_distributions')])
def test_multi_metric_search_forwards_metadata(SearchCV, param_search):
    """Test that *SearchCV forwards metadata correctly when passed multiple metrics."""
    X, y = make_classification(random_state=42)
    n_samples = _num_samples(X)
    rng = np.random.RandomState(0)
    score_weights = rng.rand(n_samples)
    score_metadata = rng.rand(n_samples)
    est = LinearSVC(dual='auto')
    param_grid_search = {param_search: {'C': [1]}}
    scorer_registry = _Registry()
    scorer = ConsumingScorer(registry=scorer_registry).set_score_request(sample_weight='score_weights', metadata='score_metadata')
    scoring = dict(my_scorer=scorer, accuracy='accuracy')
    SearchCV(est, refit='accuracy', cv=2, scoring=scoring, **param_grid_search).fit(X, y, score_weights=score_weights, score_metadata=score_metadata)
    assert len(scorer_registry)
    for _scorer in scorer_registry:
        check_recorded_metadata(obj=_scorer, method='score', split_params=('sample_weight', 'metadata'), sample_weight=score_weights, metadata=score_metadata)