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
def test_no_path():
    alphas_, _, coef_path_ = linear_model.lars_path(X, y, method='lar')
    alpha_, _, coef = linear_model.lars_path(X, y, method='lar', return_path=False)
    assert_array_almost_equal(coef, coef_path_[:, -1])
    assert alpha_ == alphas_[-1]