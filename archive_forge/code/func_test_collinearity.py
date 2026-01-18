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
def test_collinearity():
    X = np.array([[3.0, 3.0, 1.0], [2.0, 2.0, 0.0], [1.0, 1.0, 0]])
    y = np.array([1.0, 0.0, 0])
    rng = np.random.RandomState(0)
    f = ignore_warnings
    _, _, coef_path_ = f(linear_model.lars_path)(X, y, alpha_min=0.01)
    assert not np.isnan(coef_path_).any()
    residual = np.dot(X, coef_path_[:, -1]) - y
    assert (residual ** 2).sum() < 1.0
    n_samples = 10
    X = rng.rand(n_samples, 5)
    y = np.zeros(n_samples)
    _, _, coef_path_ = linear_model.lars_path(X, y, Gram='auto', copy_X=False, copy_Gram=False, alpha_min=0.0, method='lasso', verbose=0, max_iter=500)
    assert_array_almost_equal(coef_path_, np.zeros_like(coef_path_))