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
@pytest.mark.skipif(sp_version < parse_version('1.6.0'), reason='The `highs` solver is available from the 1.6.0 scipy version')
@pytest.mark.parametrize('quantile', [0.2, 0.5, 0.8])
def test_asymmetric_error(quantile, default_solver):
    """Test quantile regression for asymmetric distributed targets."""
    n_samples = 1000
    rng = np.random.RandomState(42)
    X = np.concatenate((np.abs(rng.randn(n_samples)[:, None]), -rng.randint(2, size=(n_samples, 1))), axis=1)
    intercept = 1.23
    coef = np.array([0.5, -2])
    assert np.min(X @ coef + intercept) > 0
    y = rng.exponential(scale=-(X @ coef + intercept) / np.log(1 - quantile), size=n_samples)
    model = QuantileRegressor(quantile=quantile, alpha=0, solver=default_solver).fit(X, y)
    assert model.intercept_ == approx(intercept, rel=0.2)
    assert_allclose(model.coef_, coef, rtol=0.6)
    assert_allclose(np.mean(model.predict(X) > y), quantile, atol=0.01)
    alpha = 0.01
    model.set_params(alpha=alpha).fit(X, y)
    model_coef = np.r_[model.intercept_, model.coef_]

    def func(coef):
        loss = mean_pinball_loss(y, X @ coef[1:] + coef[0], alpha=quantile)
        L1 = np.sum(np.abs(coef[1:]))
        return loss + alpha * L1
    res = minimize(fun=func, x0=[1, 0, -1], method='Nelder-Mead', tol=1e-12, options={'maxiter': 2000})
    assert func(model_coef) == approx(func(res.x))
    assert_allclose(model.intercept_, res.x[0])
    assert_allclose(model.coef_, res.x[1:])
    assert_allclose(np.mean(model.predict(X) > y), quantile, atol=0.01)