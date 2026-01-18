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
def test_lasso_lars_vs_lasso_cd_ill_conditioned2():
    X = [[1e+20, 1e+20, 0], [-1e-32, 0, 0], [1, 1, 1]]
    y = [10, 10, 1]
    alpha = 0.0001

    def objective_function(coef):
        return 1.0 / (2.0 * len(X)) * linalg.norm(y - np.dot(X, coef)) ** 2 + alpha * linalg.norm(coef, 1)
    lars = linear_model.LassoLars(alpha=alpha)
    warning_message = 'Regressors in active set degenerate.'
    with pytest.warns(ConvergenceWarning, match=warning_message):
        lars.fit(X, y)
    lars_coef_ = lars.coef_
    lars_obj = objective_function(lars_coef_)
    coord_descent = linear_model.Lasso(alpha=alpha, tol=0.0001)
    cd_coef_ = coord_descent.fit(X, y).coef_
    cd_obj = objective_function(cd_coef_)
    assert lars_obj < cd_obj * (1.0 + 1e-08)