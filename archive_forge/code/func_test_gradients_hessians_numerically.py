import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg, optimize
from sklearn._loss.loss import (
from sklearn.datasets import make_low_rank_matrix
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('base_loss', LOSSES)
@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('sample_weight', [None, 'range'])
@pytest.mark.parametrize('l2_reg_strength', [0, 1])
def test_gradients_hessians_numerically(base_loss, fit_intercept, sample_weight, l2_reg_strength):
    """Test gradients and hessians with numerical derivatives.

    Gradient should equal the numerical derivatives of the loss function.
    Hessians should equal the numerical derivatives of gradients.
    """
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    n_samples, n_features = (10, 5)
    X, y, coef = random_X_y_coef(linear_model_loss=loss, n_samples=n_samples, n_features=n_features, seed=42)
    coef = coef.ravel(order='F')
    if sample_weight == 'range':
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])
    eps = 1e-06
    g, hessp = loss.gradient_hessian_product(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    approx_g1 = optimize.approx_fprime(coef, lambda coef: loss.loss(coef - eps, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength), 2 * eps)
    approx_g2 = optimize.approx_fprime(coef, lambda coef: loss.loss(coef - 2 * eps, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength), 4 * eps)
    approx_g = (4 * approx_g1 - approx_g2) / 3
    assert_allclose(g, approx_g, rtol=0.01, atol=1e-08)
    vector = np.zeros_like(g)
    vector[1] = 1
    hess_col = hessp(vector)
    eps = 0.001
    d_x = np.linspace(-eps, eps, 30)
    d_grad = np.array([loss.gradient(coef + t * vector, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength) for t in d_x])
    d_grad -= d_grad.mean(axis=0)
    approx_hess_col = linalg.lstsq(d_x[:, np.newaxis], d_grad)[0].ravel()
    assert_allclose(approx_hess_col, hess_col, rtol=0.001)