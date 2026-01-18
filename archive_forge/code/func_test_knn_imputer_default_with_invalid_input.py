import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na', [np.nan, -1])
def test_knn_imputer_default_with_invalid_input(na):
    X = np.array([[np.inf, 1, 1, 2, na], [2, 1, 2, 2, 3], [3, 2, 3, 3, 8], [na, 6, 0, 5, 13], [na, 7, 0, 7, 8], [6, 6, 2, 5, 7]])
    with pytest.raises(ValueError, match='Input X contains (infinity|NaN)'):
        KNNImputer(missing_values=na).fit(X)
    X = np.array([[np.inf, 1, 1, 2, na], [2, 1, 2, 2, 3], [3, 2, 3, 3, 8], [na, 6, 0, 5, 13], [na, 7, 0, 7, 8], [6, 6, 2, 5, 7]])
    X_fit = np.array([[0, 1, 1, 2, na], [2, 1, 2, 2, 3], [3, 2, 3, 3, 8], [na, 6, 0, 5, 13], [na, 7, 0, 7, 8], [6, 6, 2, 5, 7]])
    imputer = KNNImputer(missing_values=na).fit(X_fit)
    with pytest.raises(ValueError, match='Input X contains (infinity|NaN)'):
        imputer.transform(X)
    imputer = KNNImputer(missing_values=0, n_neighbors=2, weights='uniform')
    X = np.array([[np.nan, 0, 0, 0, 5], [np.nan, 1, 0, np.nan, 3], [np.nan, 2, 0, 0, 0], [np.nan, 6, 0, 5, 13]])
    msg = 'Input X contains NaN'
    with pytest.raises(ValueError, match=msg):
        imputer.fit(X)
    X = np.array([[0, 0], [np.nan, 2]])