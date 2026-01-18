import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [np.nan, -1])
def test_knn_imputer_weight_uniform(na):
    X = np.array([[0, 0], [na, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]])
    X_imputed_uniform = np.array([[0, 0], [5, 2], [4, 3], [5, 6], [7, 7], [9, 8], [11, 10]])
    imputer = KNNImputer(weights='uniform', missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)

    def no_weight(dist):
        return None
    imputer = KNNImputer(weights=no_weight, missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)

    def uniform_weight(dist):
        return np.ones_like(dist)
    imputer = KNNImputer(weights=uniform_weight, missing_values=na)
    assert_allclose(imputer.fit_transform(X), X_imputed_uniform)