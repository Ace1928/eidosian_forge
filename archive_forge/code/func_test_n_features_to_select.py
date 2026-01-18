import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('direction', ('forward', 'backward'))
@pytest.mark.parametrize('n_features_to_select', (1, 5, 9, 'auto'))
def test_n_features_to_select(direction, n_features_to_select):
    n_features = 10
    X, y = make_regression(n_features=n_features, random_state=0)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features_to_select, direction=direction, cv=2)
    sfs.fit(X, y)
    if n_features_to_select == 'auto':
        n_features_to_select = n_features // 2
    assert sfs.get_support(indices=True).shape[0] == n_features_to_select
    assert sfs.n_features_to_select_ == n_features_to_select
    assert sfs.transform(X).shape[1] == n_features_to_select