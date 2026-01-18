import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
def test_knn_imputer_callable_metric():

    def custom_callable(x, y, missing_values=np.nan, squared=False):
        x = np.ma.array(x, mask=np.isnan(x))
        y = np.ma.array(y, mask=np.isnan(y))
        dist = np.nansum(np.abs(x - y))
        return dist
    X = np.array([[4, 3, 3, np.nan], [6, 9, 6, 9], [4, 8, 6, 9], [np.nan, 9, 11, 10.0]])
    X_0_3 = (9 + 9) / 2
    X_3_0 = (6 + 4) / 2
    X_imputed = np.array([[4, 3, 3, X_0_3], [6, 9, 6, 9], [4, 8, 6, 9], [X_3_0, 9, 11, 10.0]])
    imputer = KNNImputer(n_neighbors=2, metric=custom_callable)
    assert_allclose(imputer.fit_transform(X), X_imputed)