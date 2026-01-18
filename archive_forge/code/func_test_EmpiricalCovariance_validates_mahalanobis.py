import numpy as np
import pytest
from sklearn import datasets
from sklearn.covariance import (
from sklearn.covariance._shrunk_covariance import _ledoit_wolf
from sklearn.utils._testing import (
from .._shrunk_covariance import _oas
def test_EmpiricalCovariance_validates_mahalanobis():
    """Checks that EmpiricalCovariance validates data with mahalanobis."""
    cov = EmpiricalCovariance().fit(X)
    msg = f'X has 2 features, but \\w+ is expecting {X.shape[1]} features as input'
    with pytest.raises(ValueError, match=msg):
        cov.mahalanobis(X[:, :2])