import itertools
import warnings
from functools import partial
import numpy as np
import pytest
import scipy
from numpy.testing import assert_allclose
from scipy import linalg
from scipy.optimize import minimize, root
from sklearn._loss import HalfBinomialLoss, HalfPoissonLoss, HalfTweedieLoss
from sklearn._loss.link import IdentityLink, LogLink
from sklearn.base import clone
from sklearn.datasets import make_low_rank_matrix, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._glm import _GeneralizedLinearRegressor
from sklearn.linear_model._glm._newton_solver import NewtonCholeskySolver
from sklearn.linear_model._linear_loss import LinearModelLoss
from sklearn.metrics import d2_tweedie_score, mean_poisson_deviance
from sklearn.model_selection import train_test_split
@pytest.mark.parametrize('fit_intercept', [False, True])
@pytest.mark.parametrize('alpha', [0.0, 1.0])
@pytest.mark.parametrize('GLMEstimator', [_GeneralizedLinearRegressor, PoissonRegressor, GammaRegressor])
def test_glm_sample_weight_consistency(fit_intercept, alpha, GLMEstimator):
    """Test that the impact of sample_weight is consistent"""
    rng = np.random.RandomState(0)
    n_samples, n_features = (10, 5)
    X = rng.rand(n_samples, n_features)
    y = rng.rand(n_samples)
    glm_params = dict(alpha=alpha, fit_intercept=fit_intercept)
    glm = GLMEstimator(**glm_params).fit(X, y)
    coef = glm.coef_.copy()
    sample_weight = np.ones(y.shape)
    glm.fit(X, y, sample_weight=sample_weight)
    assert_allclose(glm.coef_, coef, rtol=1e-12)
    sample_weight = 2 * np.ones(y.shape)
    glm.fit(X, y, sample_weight=sample_weight)
    assert_allclose(glm.coef_, coef, rtol=1e-12)
    sample_weight = np.ones(y.shape)
    sample_weight[-1] = 0
    glm.fit(X, y, sample_weight=sample_weight)
    coef1 = glm.coef_.copy()
    glm.fit(X[:-1], y[:-1])
    assert_allclose(glm.coef_, coef1, rtol=1e-12)
    X2 = np.concatenate([X, X[:n_samples // 2]], axis=0)
    y2 = np.concatenate([y, y[:n_samples // 2]])
    sample_weight_1 = np.ones(len(y))
    sample_weight_1[:n_samples // 2] = 2
    glm1 = GLMEstimator(**glm_params).fit(X, y, sample_weight=sample_weight_1)
    glm2 = GLMEstimator(**glm_params).fit(X2, y2, sample_weight=None)
    assert_allclose(glm1.coef_, glm2.coef_)