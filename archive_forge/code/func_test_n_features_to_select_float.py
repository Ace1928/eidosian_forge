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
@pytest.mark.parametrize('n_features_to_select, expected', ((0.1, 1), (1.0, 10), (0.5, 5)))
def test_n_features_to_select_float(direction, n_features_to_select, expected):
    X, y = make_regression(n_features=10)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features_to_select, direction=direction, cv=2)
    sfs.fit(X, y)
    assert sfs.n_features_to_select_ == expected