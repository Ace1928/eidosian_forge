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
def test_linalg_warning_with_newton_solver(global_random_seed):
    newton_solver = 'newton-cholesky'
    rng = np.random.RandomState(global_random_seed)
    X_orig = rng.normal(size=(20, 3))
    y = rng.poisson(np.exp(X_orig @ np.ones(X_orig.shape[1])), size=X_orig.shape[0]).astype(np.float64)
    X_collinear = np.hstack([X_orig] * 10)
    baseline_pred = np.full_like(y, y.mean())
    constant_model_deviance = mean_poisson_deviance(y, baseline_pred)
    assert constant_model_deviance > 1.0
    tol = 1e-10
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        reg = PoissonRegressor(solver=newton_solver, alpha=0.0, tol=tol).fit(X_orig, y)
    original_newton_deviance = mean_poisson_deviance(y, reg.predict(X_orig))
    assert original_newton_deviance > 0.2
    assert constant_model_deviance - original_newton_deviance > 0.1
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        reg = PoissonRegressor(solver='lbfgs', alpha=0.0, tol=tol).fit(X_collinear, y)
    collinear_lbfgs_deviance = mean_poisson_deviance(y, reg.predict(X_collinear))
    rtol = 1e-06
    assert collinear_lbfgs_deviance == pytest.approx(original_newton_deviance, rel=rtol)
    msg = 'The inner solver of .*Newton.*Solver stumbled upon a singular or very ill-conditioned Hessian matrix'
    with pytest.warns(scipy.linalg.LinAlgWarning, match=msg):
        reg = PoissonRegressor(solver=newton_solver, alpha=0.0, tol=tol).fit(X_collinear, y)
    collinear_newton_deviance = mean_poisson_deviance(y, reg.predict(X_collinear))
    assert collinear_newton_deviance == pytest.approx(original_newton_deviance, rel=rtol)
    with warnings.catch_warnings():
        warnings.simplefilter('error', scipy.linalg.LinAlgWarning)
        reg = PoissonRegressor(solver=newton_solver, alpha=1e-10).fit(X_collinear, y)
    penalized_collinear_newton_deviance = mean_poisson_deviance(y, reg.predict(X_collinear))
    assert penalized_collinear_newton_deviance == pytest.approx(original_newton_deviance, rel=rtol)