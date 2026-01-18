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
def test_n_features_to_select_auto(direction):
    """Check the behaviour of `n_features_to_select="auto"` with different
    values for the parameter `tol`.
    """
    n_features = 10
    tol = 0.001
    X, y = make_regression(n_features=n_features, random_state=0)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select='auto', tol=tol, direction=direction, cv=2)
    sfs.fit(X, y)
    max_features_to_select = n_features - 1
    assert sfs.get_support(indices=True).shape[0] <= max_features_to_select
    assert sfs.n_features_to_select_ <= max_features_to_select
    assert sfs.transform(X).shape[1] <= max_features_to_select
    assert sfs.get_support(indices=True).shape[0] == sfs.n_features_to_select_