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
@pytest.mark.parametrize('y_array_constr', [np.array] + CSR_CONTAINERS, ids=['dense'] + [container.__name__ for container in CSR_CONTAINERS])
def test_euclidean_distances_with_norms(global_dtype, y_array_constr):
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 10)).astype(global_dtype, copy=False)
    Y = rng.random_sample((20, 10)).astype(global_dtype, copy=False)
    X_norm_sq = (X.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)
    Y_norm_sq = (Y.astype(np.float64) ** 2).sum(axis=1).reshape(1, -1)
    Y = y_array_constr(Y)
    D1 = euclidean_distances(X, Y)
    D2 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq)
    D3 = euclidean_distances(X, Y, Y_norm_squared=Y_norm_sq)
    D4 = euclidean_distances(X, Y, X_norm_squared=X_norm_sq, Y_norm_squared=Y_norm_sq)
    assert_allclose(D2, D1)
    assert_allclose(D3, D1)
    assert_allclose(D4, D1)
    wrong_D = euclidean_distances(X, Y, X_norm_squared=np.zeros_like(X_norm_sq), Y_norm_squared=np.zeros_like(Y_norm_sq))
    with pytest.raises(AssertionError):
        assert_allclose(wrong_D, D1)