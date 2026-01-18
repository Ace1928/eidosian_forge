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
def test_lars_path_positive_constraint():
    err_msg = "Positive constraint not supported for 'lar' coding method."
    with pytest.raises(ValueError, match=err_msg):
        linear_model.lars_path(diabetes['data'], diabetes['target'], method='lar', positive=True)
    method = 'lasso'
    _, _, coefs = linear_model.lars_path(X, y, return_path=True, method=method, positive=False)
    assert coefs.min() < 0
    _, _, coefs = linear_model.lars_path(X, y, return_path=True, method=method, positive=True)
    assert coefs.min() >= 0