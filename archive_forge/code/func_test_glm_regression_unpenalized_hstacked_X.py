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
@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('fit_intercept', [True, False])
def test_glm_regression_unpenalized_hstacked_X(solver, fit_intercept, glm_dataset):
    """Test that unpenalized GLM converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    GLM fit on [X] is the same as fit on [X, X]/2.
    For long X, [X, X] is a singular matrix and we check against the minimum norm
    solution:
        min ||w||_2 subject to w = argmin deviance(X, y, w)
    """
    model, X, y, coef, _, _, _ = glm_dataset
    n_samples, n_features = X.shape
    alpha = 0
    params = dict(alpha=alpha, fit_intercept=fit_intercept, solver=solver, tol=1e-12, max_iter=1000)
    model = clone(model).set_params(**params)
    if fit_intercept:
        intercept = coef[-1]
        coef = coef[:-1]
        if n_samples > n_features:
            X = X[:, :-1]
            X = 0.5 * np.concatenate((X, X), axis=1)
        else:
            X = np.c_[X[:, :-1], X[:, :-1], X[:, -1]]
    else:
        intercept = 0
        X = 0.5 * np.concatenate((X, X), axis=1)
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)
    with warnings.catch_warnings():
        if solver.startswith('newton'):
            warnings.filterwarnings('ignore', category=scipy.linalg.LinAlgWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model.fit(X, y)
    if fit_intercept and n_samples < n_features:
        model_intercept = 2 * model.intercept_
        model_coef = 2 * model.coef_[:-1]
    else:
        model_intercept = model.intercept_
        model_coef = model.coef_
    if n_samples > n_features:
        assert model_intercept == pytest.approx(intercept)
        rtol = 0.0001
        assert_allclose(model_coef, np.r_[coef, coef], rtol=rtol)
    else:
        rtol = 1e-06 if solver == 'lbfgs' else 5e-06
        assert_allclose(model.predict(X), y, rtol=rtol)
        if solver == 'lbfgs' and fit_intercept or solver == 'newton-cholesky':
            norm_solution = np.linalg.norm(0.5 * np.r_[intercept, intercept, coef, coef])
            norm_model = np.linalg.norm(np.r_[model.intercept_, model.coef_])
            assert norm_model > (1 + 1e-12) * norm_solution
        else:
            assert model_intercept == pytest.approx(intercept, rel=5e-06)
            assert_allclose(model_coef, np.r_[coef, coef], rtol=0.0001)