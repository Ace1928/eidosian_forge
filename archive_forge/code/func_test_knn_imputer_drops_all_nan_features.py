import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [-1, np.nan])
def test_knn_imputer_drops_all_nan_features(na):
    X1 = np.array([[na, 1], [na, 2]])
    knn = KNNImputer(missing_values=na, n_neighbors=1)
    X1_expected = np.array([[1], [2]])
    assert_allclose(knn.fit_transform(X1), X1_expected)
    X2 = np.array([[1, 2], [3, na]])
    X2_expected = np.array([[2], [1.5]])
    assert_allclose(knn.transform(X2), X2_expected)