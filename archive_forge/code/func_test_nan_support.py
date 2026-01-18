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
def test_nan_support():
    rng = np.random.RandomState(0)
    n_samples, n_features = (40, 4)
    X, y = make_regression(n_samples, n_features, random_state=0)
    nan_mask = rng.randint(0, 2, size=(n_samples, n_features), dtype=bool)
    X[nan_mask] = np.nan
    sfs = SequentialFeatureSelector(HistGradientBoostingRegressor(), n_features_to_select='auto', cv=2)
    sfs.fit(X, y)
    sfs.transform(X)
    with pytest.raises(ValueError, match='Input X contains NaN'):
        SequentialFeatureSelector(LinearRegression(), n_features_to_select='auto', cv=2).fit(X, y)