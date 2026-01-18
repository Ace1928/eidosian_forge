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
@pytest.mark.parametrize('fit_intercept', [False, True])
def test_glm_regression(solver, fit_intercept, glm_dataset):
    """Test that GLM converges for all solvers to correct solution.

    We work with a simple constructed data set with known solution.
    """
    model, X, y, _, coef_with_intercept, coef_without_intercept, alpha = glm_dataset
    params = dict(alpha=alpha, fit_intercept=fit_intercept, solver=solver, tol=1e-12, max_iter=1000)
    model = clone(model).set_params(**params)
    X = X[:, :-1]
    if fit_intercept:
        coef = coef_with_intercept
        intercept = coef[-1]
        coef = coef[:-1]
    else:
        coef = coef_without_intercept
        intercept = 0
    model.fit(X, y)
    rtol = 5e-05 if solver == 'lbfgs' else 1e-09
    assert model.intercept_ == pytest.approx(intercept, rel=rtol)
    assert_allclose(model.coef_, coef, rtol=rtol)
    model = clone(model).set_params(**params).fit(X, y, sample_weight=np.ones(X.shape[0]))
    assert model.intercept_ == pytest.approx(intercept, rel=rtol)
    assert_allclose(model.coef_, coef, rtol=rtol)