import copy
import pickle
import warnings
import numpy as np
import pytest
from scipy.special import expit
import sklearn
from sklearn.datasets import make_regression
from sklearn.isotonic import (
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.validation import check_array
@pytest.mark.parametrize('increasing', [True, False])
def test_isotonic_thresholds(increasing):
    rng = np.random.RandomState(42)
    n_samples = 30
    X = rng.normal(size=n_samples)
    y = rng.normal(size=n_samples)
    ireg = IsotonicRegression(increasing=increasing).fit(X, y)
    X_thresholds, y_thresholds = (ireg.X_thresholds_, ireg.y_thresholds_)
    assert X_thresholds.shape == y_thresholds.shape
    assert X_thresholds.shape[0] < X.shape[0]
    assert np.isin(X_thresholds, X).all()
    assert y_thresholds.max() <= y.max()
    assert y_thresholds.min() >= y.min()
    assert all(np.diff(X_thresholds) > 0)
    if increasing:
        assert all(np.diff(y_thresholds) >= 0)
    else:
        assert all(np.diff(y_thresholds) <= 0)