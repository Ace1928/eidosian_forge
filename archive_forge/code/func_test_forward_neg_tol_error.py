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
def test_forward_neg_tol_error():
    """Check that we raise an error when tol<0 and direction='forward'"""
    X, y = make_regression(n_features=10, random_state=0)
    sfs = SequentialFeatureSelector(LinearRegression(), n_features_to_select='auto', direction='forward', tol=-0.001)
    with pytest.raises(ValueError, match='tol must be positive'):
        sfs.fit(X, y)