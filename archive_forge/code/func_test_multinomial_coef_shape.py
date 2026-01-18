import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, optimize
from sklearn._loss.loss import (
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('fit_intercept', [False, True])
def test_multinomial_coef_shape(fit_intercept):
    """Test that multinomial LinearModelLoss respects shape of coef."""
    loss = LinearModelLoss(base_loss=HalfMultinomialLoss(), fit_intercept=fit_intercept)
    n_samples, n_features = (10, 5)
    X, y, coef = random_X_y_coef(linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42)
    s = np.random.RandomState(42).randn(*coef.shape)
    l, g = loss.loss_gradient(coef, X, y)
    g1 = loss.gradient(coef, X, y)
    g2, hessp = loss.gradient_hessian_product(coef, X, y)
    h = hessp(s)
    assert g.shape == coef.shape
    assert h.shape == coef.shape
    assert_allclose(g, g1)
    assert_allclose(g, g2)
    coef_r = coef.ravel(order='F')
    s_r = s.ravel(order='F')
    l_r, g_r = loss.loss_gradient(coef_r, X, y)
    g1_r = loss.gradient(coef_r, X, y)
    g2_r, hessp_r = loss.gradient_hessian_product(coef_r, X, y)
    h_r = hessp_r(s_r)
    assert g_r.shape == coef_r.shape
    assert h_r.shape == coef_r.shape
    assert_allclose(g_r, g1_r)
    assert_allclose(g_r, g2_r)
    assert_allclose(g, g_r.reshape(loss.base_loss.n_classes, -1, order='F'))
    assert_allclose(h, h_r.reshape(loss.base_loss.n_classes, -1, order='F'))