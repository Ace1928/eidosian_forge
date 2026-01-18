import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('working_memory', [None, 0])
@pytest.mark.parametrize('na', [-1, np.nan])
def test_knn_imputer_distance_weighted_not_enough_neighbors(na, working_memory):
    X = np.array([[3, na], [2, na], [na, 4], [5, 6], [6, 8], [na, 5]])
    dist = pairwise_distances(X, metric='nan_euclidean', squared=False, missing_values=na)
    X_01 = np.average(X[3:5, 1], weights=1 / dist[0, 3:5])
    X_11 = np.average(X[3:5, 1], weights=1 / dist[1, 3:5])
    X_20 = np.average(X[3:5, 0], weights=1 / dist[2, 3:5])
    X_50 = np.average(X[3:5, 0], weights=1 / dist[5, 3:5])
    X_expected = np.array([[3, X_01], [2, X_11], [X_20, 4], [5, 6], [6, 8], [X_50, 5]])
    with config_context(working_memory=working_memory):
        knn_3 = KNNImputer(missing_values=na, n_neighbors=3, weights='distance')
        assert_allclose(knn_3.fit_transform(X), X_expected)
        knn_4 = KNNImputer(missing_values=na, n_neighbors=4, weights='distance')
        assert_allclose(knn_4.fit_transform(X), X_expected)