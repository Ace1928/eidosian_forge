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
@pytest.mark.parametrize('copy_X', [True, False])
def test_lasso_lars_copyX_behaviour(copy_X):
    """
    Test that user input regarding copy_X is not being overridden (it was until
    at least version 0.21)

    """
    lasso_lars = LassoLarsIC(copy_X=copy_X, precompute=False)
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 5))
    X_copy = X.copy()
    y = X[:, 2]
    lasso_lars.fit(X, y)
    assert copy_X == np.array_equal(X, X_copy)