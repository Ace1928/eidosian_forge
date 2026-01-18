import numpy as np
import pytest
from pytest import approx
from scipy.optimize import minimize
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import HuberRegressor, QuantileRegressor
from sklearn.metrics import mean_pinball_loss
from sklearn.utils._testing import assert_allclose, skip_if_32bit
from sklearn.utils.fixes import (
def test_error_interior_point_future(X_y_data, monkeypatch):
    """Check that we will raise a proper error when requesting
    `solver='interior-point'` in SciPy >= 1.11.
    """
    X, y = X_y_data
    import sklearn.linear_model._quantile
    with monkeypatch.context() as m:
        m.setattr(sklearn.linear_model._quantile, 'sp_version', parse_version('1.11.0'))
        err_msg = 'Solver interior-point is not anymore available in SciPy >= 1.11.0.'
        with pytest.raises(ValueError, match=err_msg):
            QuantileRegressor(solver='interior-point').fit(X, y)