import warnings
import numpy as np
import pytest
from scipy import linalg
from sklearn import datasets, linear_model
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._least_angle import _lars_path_residues
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import (
def test_lasso_lars_vs_lasso_cd():
    X = 3 * diabetes.data
    alphas, _, lasso_path = linear_model.lars_path(X, y, method='lasso')
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08)
    for c, a in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01
    for alpha in np.linspace(0.01, 1 - 0.01, 20):
        clf1 = linear_model.LassoLars(alpha=alpha).fit(X, y)
        clf2 = linear_model.Lasso(alpha=alpha, tol=1e-08).fit(X, y)
        err = linalg.norm(clf1.coef_ - clf2.coef_)
        assert err < 0.001
    X = diabetes.data
    X = X - X.sum(axis=0)
    X /= np.linalg.norm(X, axis=0)
    alphas, _, lasso_path = linear_model.lars_path(X, y, method='lasso')
    lasso_cd = linear_model.Lasso(fit_intercept=False, tol=1e-08)
    for c, a in zip(lasso_path.T, alphas):
        if a == 0:
            continue
        lasso_cd.alpha = a
        lasso_cd.fit(X, y)
        error = linalg.norm(c - lasso_cd.coef_)
        assert error < 0.01