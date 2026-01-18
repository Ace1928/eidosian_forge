import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [np.nan, -1])
def test_knn_imputer_removes_all_na_features(na):
    X = np.array([[1, 1, na, 1, 1, 1.0], [2, 3, na, 2, 2, 2], [3, 4, na, 3, 3, na], [6, 4, na, na, 6, 6]])
    knn = KNNImputer(missing_values=na, n_neighbors=2).fit(X)
    X_transform = knn.transform(X)
    assert not np.isnan(X_transform).any()
    assert X_transform.shape == (4, 5)
    X_test = np.arange(0, 12).reshape(2, 6)
    X_transform = knn.transform(X_test)
    assert_allclose(X_test[:, [0, 1, 3, 4, 5]], X_transform)