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
def test_lasso_lars_vs_lasso_cd_ill_conditioned():
    rng = np.random.RandomState(42)
    n, m = (70, 100)
    k = 5
    X = rng.randn(n, m)
    w = np.zeros((m, 1))
    i = np.arange(0, m)
    rng.shuffle(i)
    supp = i[:k]
    w[supp] = np.sign(rng.randn(k, 1)) * (rng.rand(k, 1) + 1)
    y = np.dot(X, w)
    sigma = 0.2
    y += sigma * rng.rand(*y.shape)
    y = y.squeeze()
    lars_alphas, _, lars_coef = linear_model.lars_path(X, y, method='lasso')
    _, lasso_coef2, _ = linear_model.lasso_path(X, y, alphas=lars_alphas, tol=1e-06)
    assert_array_almost_equal(lars_coef, lasso_coef2, decimal=1)