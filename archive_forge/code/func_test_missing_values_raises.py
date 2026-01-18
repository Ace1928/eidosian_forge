import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
from sklearn.tree import (
from sklearn.utils._testing import assert_allclose
from sklearn.utils.fixes import CSC_CONTAINERS
@pytest.mark.parametrize('DecisionTreeEstimator', [DecisionTreeClassifier, DecisionTreeRegressor])
def test_missing_values_raises(DecisionTreeEstimator):
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, n_informative=3, random_state=0)
    X[0, 0] = np.nan
    monotonic_cst = np.zeros(X.shape[1])
    monotonic_cst[0] = 1
    est = DecisionTreeEstimator(max_depth=None, monotonic_cst=monotonic_cst, random_state=0)
    msg = 'Input X contains NaN'
    with pytest.raises(ValueError, match=msg):
        est.fit(X, y)