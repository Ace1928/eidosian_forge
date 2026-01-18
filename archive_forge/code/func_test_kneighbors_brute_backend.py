import re
import warnings
from itertools import product
import joblib
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn import (
from sklearn.base import clone
from sklearn.exceptions import DataConversionWarning, EfficiencyWarning, NotFittedError
from sklearn.metrics._dist_metrics import (
from sklearn.metrics.pairwise import PAIRWISE_BOOLEAN_FUNCTIONS, pairwise_distances
from sklearn.metrics.tests.test_dist_metrics import BOOL_METRICS
from sklearn.metrics.tests.test_pairwise_distances_reduction import (
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import (
from sklearn.neighbors._base import (
from sklearn.pipeline import make_pipeline
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.validation import check_random_state
@pytest.mark.parametrize('metric', sorted(set(neighbors.VALID_METRICS['brute']) - set(['precomputed'])))
def test_kneighbors_brute_backend(metric, global_dtype, global_random_seed, n_samples=2000, n_features=30, n_query_pts=5, n_neighbors=5):
    rng = np.random.RandomState(global_random_seed)
    X_train = rng.rand(n_samples, n_features).astype(global_dtype, copy=False)
    X_test = rng.rand(n_query_pts, n_features).astype(global_dtype, copy=False)
    if metric == 'haversine':
        feature_sl = slice(None, 2)
        X_train = np.ascontiguousarray(X_train[:, feature_sl])
        X_test = np.ascontiguousarray(X_test[:, feature_sl])
    if metric in PAIRWISE_BOOLEAN_FUNCTIONS:
        X_train = X_train > 0.5
        X_test = X_test > 0.5
    metric_params_list = _generate_test_params_for(metric, n_features)
    for metric_params in metric_params_list:
        p = metric_params.pop('p', 2)
        neigh = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute', metric=metric, p=p, metric_params=metric_params)
        neigh.fit(X_train)
        with config_context(enable_cython_pairwise_dist=False):
            legacy_brute_dst, legacy_brute_idx = neigh.kneighbors(X_test, return_distance=True)
        with config_context(enable_cython_pairwise_dist=True):
            pdr_brute_dst, pdr_brute_idx = neigh.kneighbors(X_test, return_distance=True)
        assert_compatible_argkmin_results(legacy_brute_dst, pdr_brute_dst, legacy_brute_idx, pdr_brute_idx)