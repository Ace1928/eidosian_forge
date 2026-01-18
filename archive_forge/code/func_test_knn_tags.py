import numpy as np
import pytest
from sklearn import config_context
from sklearn.impute import KNNImputer
from sklearn.metrics.pairwise import nan_euclidean_distances, pairwise_distances
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils._testing import assert_allclose
@pytest.mark.parametrize('na, allow_nan', [(-1, False), (np.nan, True)])
def test_knn_tags(na, allow_nan):
    knn = KNNImputer(missing_values=na)
    assert knn._get_tags()['allow_nan'] == allow_nan