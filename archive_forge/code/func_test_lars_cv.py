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
def test_lars_cv():
    old_alpha = 0
    lars_cv = linear_model.LassoLarsCV()
    for length in (400, 200, 100):
        X = diabetes.data[:length]
        y = diabetes.target[:length]
        lars_cv.fit(X, y)
        np.testing.assert_array_less(old_alpha, lars_cv.alpha_)
        old_alpha = lars_cv.alpha_
    assert not hasattr(lars_cv, 'n_nonzero_coefs')