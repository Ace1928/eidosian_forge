from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import optimize
from scipy.special import factorial, xlogy
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
from sklearn.metrics._regression import _check_reg_targets
from sklearn.model_selection import GridSearchCV
from sklearn.utils._testing import (
@pytest.mark.parametrize('distribution', ['normal', 'lognormal', 'exponential', 'uniform'])
@pytest.mark.parametrize('target_quantile', [0.05, 0.5, 0.75])
def test_mean_pinball_loss_on_constant_predictions(distribution, target_quantile):
    if not hasattr(np, 'quantile'):
        pytest.skip('This test requires a more recent version of numpy with support for np.quantile.')
    n_samples = 3000
    rng = np.random.RandomState(42)
    data = getattr(rng, distribution)(size=n_samples)
    best_pred = np.quantile(data, target_quantile)
    best_constant_pred = np.full(n_samples, fill_value=best_pred)
    best_pbl = mean_pinball_loss(data, best_constant_pred, alpha=target_quantile)
    candidate_predictions = np.quantile(data, np.linspace(0, 1, 100))
    for pred in candidate_predictions:
        constant_pred = np.full(n_samples, fill_value=pred)
        pbl = mean_pinball_loss(data, constant_pred, alpha=target_quantile)
        assert pbl >= best_pbl - np.finfo(best_pbl.dtype).eps
        expected_pbl = (pred - data[data < pred]).sum() * (1 - target_quantile) + (data[data >= pred] - pred).sum() * target_quantile
        expected_pbl /= n_samples
        assert_almost_equal(expected_pbl, pbl)

    def objective_func(x):
        constant_pred = np.full(n_samples, fill_value=x)
        return mean_pinball_loss(data, constant_pred, alpha=target_quantile)
    result = optimize.minimize(objective_func, data.mean(), method='Nelder-Mead')
    assert result.success
    assert result.x == pytest.approx(best_pred, rel=0.01)
    assert result.fun == pytest.approx(best_pbl)