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
def test_bad_n_features_to_select():
    n_features = 5
    X, y = make_regression(n_features=n_features)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select=n_features)
    with pytest.raises(ValueError, match='n_features_to_select must be < n_features'):
        sfs.fit(X, y)