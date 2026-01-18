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
def test_glm_regression_unpenalized_vstacked_X(solver, fit_intercept, glm_dataset):
    """Test that unpenalized GLM converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    GLM fit on [X] is the same as fit on [X], [y]
                                         [X], [y].
    For wide X, [X', X'] is a singular matrix and we check against the minimum norm
    solution:
        min ||w||_2 subject to w = argmin deviance(X, y, w)
    """
    model, X, y, coef, _, _, _ = glm_dataset
    n_samples, n_features = X.shape
    alpha = 0
    params = dict(alpha=alpha, fit_intercept=fit_intercept, solver=solver, tol=1e-12, max_iter=1000)
    model = clone(model).set_params(**params)
    if fit_intercept:
        X = X[:, :-1]
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        intercept = 0
    X = np.concatenate((X, X), axis=0)
    assert np.linalg.matrix_rank(X) <= min(n_samples, n_features)
    y = np.r_[y, y]
    with warnings.catch_warnings():
        if solver.startswith('newton') and n_samples < n_features:
            warnings.filterwarnings('ignore', category=scipy.linalg.LinAlgWarning)
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model.fit(X, y)
    if n_samples > n_features:
        rtol = 5e-05 if solver == 'lbfgs' else 1e-06
        assert model.intercept_ == pytest.approx(intercept)
        assert_allclose(model.coef_, coef, rtol=rtol)
    else:
        rtol = 1e-06 if solver == 'lbfgs' else 5e-06
        assert_allclose(model.predict(X), y, rtol=rtol)
        norm_solution = np.linalg.norm(np.r_[intercept, coef])
        norm_model = np.linalg.norm(np.r_[model.intercept_, model.coef_])
        if solver == 'newton-cholesky':
            if not norm_model > (1 + 1e-12) * norm_solution:
                assert model.intercept_ == pytest.approx(intercept)
                assert_allclose(model.coef_, coef, rtol=0.0001)
        elif solver == 'lbfgs' and fit_intercept:
            assert norm_model > (1 + 1e-12) * norm_solution
        else:
            rtol = 1e-05 if solver == 'newton-cholesky' else 0.0001
            assert model.intercept_ == pytest.approx(intercept, rel=rtol)
            assert_allclose(model.coef_, coef, rtol=rtol)