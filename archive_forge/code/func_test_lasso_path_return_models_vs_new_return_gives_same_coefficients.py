import warnings
from copy import deepcopy
import joblib
import numpy as np
import pytest
from scipy import interpolate, sparse
from sklearn.base import clone, is_classifier
from sklearn.datasets import load_diabetes, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import (
from sklearn.linear_model._coordinate_descent import _set_order
from sklearn.model_selection import (
from sklearn.model_selection._split import GroupsConsumerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSC_CONTAINERS, CSR_CONTAINERS
def test_lasso_path_return_models_vs_new_return_gives_same_coefficients():
    X = np.array([[1, 2, 3.1], [2.3, 5.4, 4.3]]).T
    y = np.array([1, 2, 3.1])
    alphas = [5.0, 1.0, 0.5]
    alphas_lars, _, coef_path_lars = lars_path(X, y, method='lasso')
    coef_path_cont_lars = interpolate.interp1d(alphas_lars[::-1], coef_path_lars[:, ::-1])
    alphas_lasso2, coef_path_lasso2, _ = lasso_path(X, y, alphas=alphas)
    coef_path_cont_lasso = interpolate.interp1d(alphas_lasso2[::-1], coef_path_lasso2[:, ::-1])
    assert_array_almost_equal(coef_path_cont_lasso(alphas), coef_path_cont_lars(alphas), decimal=1)