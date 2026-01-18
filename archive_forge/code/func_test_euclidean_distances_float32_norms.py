import warnings
from types import GeneratorType
import numpy as np
from numpy import linalg
from scipy.sparse import issparse
from scipy.spatial.distance import (
import pytest
from sklearn import config_context
from sklearn.exceptions import DataConversionWarning
from sklearn.metrics.pairwise import (
from sklearn.preprocessing import normalize
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
from sklearn.utils.parallel import Parallel, delayed
@pytest.mark.parametrize('symmetric', [True, False])
def test_euclidean_distances_float32_norms(global_random_seed, symmetric):
    rng = np.random.RandomState(global_random_seed)
    X = rng.random_sample((10, 10))
    Y = X if symmetric else rng.random_sample((20, 10))
    X_norm_sq = (X.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)
    Y_norm_sq = (Y.astype(np.float32) ** 2).sum(axis=1).reshape(1, -1)
    D1 = euclidean_distances(X, Y)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    assert_allclose(D4, D1)