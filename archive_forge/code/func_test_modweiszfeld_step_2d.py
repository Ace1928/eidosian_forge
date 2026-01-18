import os
import re
import sys
from contextlib import contextmanager
import numpy as np
import pytest
from numpy.testing import (
from scipy.linalg import norm
from scipy.optimize import fmin_bfgs
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from sklearn.linear_model._theil_sen import (
from sklearn.utils._testing import assert_almost_equal
def test_modweiszfeld_step_2d():
    X = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0]).reshape(3, 2)
    y = np.array([0.5, 0.5])
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_almost_equal(new_y, np.array([1 / 3, 2 / 3]))
    new_y = _modified_weiszfeld_step(X, new_y)
    assert_array_almost_equal(new_y, np.array([0.2792408, 0.7207592]))
    y = np.array([0.21132505, 0.78867497])
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_almost_equal(new_y, y)