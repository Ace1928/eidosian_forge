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
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_loss_grad_hess_are_the_same(base_loss, fit_intercept, sample_weight, l2_reg_strength, csr_container):
    """Test that loss and gradient are the same across different functions."""
    loss = LinearModelLoss(base_loss=base_loss(), fit_intercept=fit_intercept)
    X, y, coef = random_X_y_coef(linear_model_loss=loss, n_samples=10, n_features=5, seed=42)
    if sample_weight == 'range':
        sample_weight = np.linspace(1, y.shape[0], num=y.shape[0])
    l1 = loss.loss(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    g1 = loss.gradient(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    l2, g2 = loss.loss_gradient(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    g3, h3 = loss.gradient_hessian_product(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    if not base_loss.is_multiclass:
        g4, h4, _ = loss.gradient_hessian(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    else:
        with pytest.raises(NotImplementedError):
            loss.gradient_hessian(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    assert_allclose(l1, l2)
    assert_allclose(g1, g2)
    assert_allclose(g1, g3)
    if not base_loss.is_multiclass:
        assert_allclose(g1, g4)
        assert_allclose(h4 @ g4, h3(g3))
    X = csr_container(X)
    l1_sp = loss.loss(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    g1_sp = loss.gradient(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    l2_sp, g2_sp = loss.loss_gradient(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    g3_sp, h3_sp = loss.gradient_hessian_product(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    if not base_loss.is_multiclass:
        g4_sp, h4_sp, _ = loss.gradient_hessian(coef, X, y, sample_weight=sample_weight, l2_reg_strength=l2_reg_strength)
    assert_allclose(l1, l1_sp)
    assert_allclose(l1, l2_sp)
    assert_allclose(g1, g1_sp)
    assert_allclose(g1, g2_sp)
    assert_allclose(g1, g3_sp)
    assert_allclose(h3(g1), h3_sp(g1_sp))
    if not base_loss.is_multiclass:
        assert_allclose(g1, g4_sp)
        assert_allclose(h4 @ g4, h4_sp @ g1_sp)