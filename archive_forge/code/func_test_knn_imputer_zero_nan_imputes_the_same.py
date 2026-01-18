import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [np.nan, -1])
def test_knn_imputer_zero_nan_imputes_the_same(na):
    X_zero = np.array([[1, 0, 1, 1, 1.0], [2, 2, 2, 2, 2], [3, 3, 3, 3, 0], [6, 6, 0, 6, 6]])
    X_nan = np.array([[1, na, 1, 1, 1.0], [2, 2, 2, 2, 2], [3, 3, 3, 3, na], [6, 6, na, 6, 6]])
    X_imputed = np.array([[1, 2.5, 1, 1, 1.0], [2, 2, 2, 2, 2], [3, 3, 3, 3, 1.5], [6, 6, 2.5, 6, 6]])
    imputer_zero = KNNImputer(missing_values=0, n_neighbors=2, weights='uniform')
    imputer_nan = KNNImputer(missing_values=na, n_neighbors=2, weights='uniform')
    assert_allclose(imputer_zero.fit_transform(X_zero), X_imputed)
    assert_allclose(imputer_zero.fit_transform(X_zero), imputer_nan.fit_transform(X_nan))