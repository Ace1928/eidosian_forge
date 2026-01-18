import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('weights', ['uniform', 'distance'])
@pytest.mark.parametrize('n_neighbors', range(1, 6))
def test_knn_imputer_shape(weights, n_neighbors):
    n_rows = 10
    n_cols = 2
    X = np.random.rand(n_rows, n_cols)
    X[0, 0] = np.nan
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    X_imputed = imputer.fit_transform(X)
    assert X_imputed.shape == (n_rows, n_cols)