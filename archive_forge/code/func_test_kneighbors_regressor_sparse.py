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
@pytest.mark.filterwarnings('ignore:EfficiencyWarning')
def test_kneighbors_regressor_sparse(n_samples=40, n_features=5, n_test_pts=10, n_neighbors=5, random_state=0):
    rng = np.random.RandomState(random_state)
    X = 2 * rng.rand(n_samples, n_features) - 1
    y = ((X ** 2).sum(axis=1) < 0.25).astype(int)
    for sparsemat in SPARSE_TYPES:
        knn = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, algorithm='auto')
        knn.fit(sparsemat(X), y)
        knn_pre = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors, metric='precomputed')
        knn_pre.fit(pairwise_distances(X, metric='euclidean'), y)
        for sparsev in SPARSE_OR_DENSE:
            X2 = sparsev(X)
            assert np.mean(knn.predict(X2).round() == y) > 0.95
            X2_pre = sparsev(pairwise_distances(X, metric='euclidean'))
            if sparsev in DOK_CONTAINERS + BSR_CONTAINERS:
                msg = 'not supported due to its handling of explicit zeros'
                with pytest.raises(TypeError, match=msg):
                    knn_pre.predict(X2_pre)
            else:
                assert np.mean(knn_pre.predict(X2_pre).round() == y) > 0.95