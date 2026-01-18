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
def test_modweiszfeld_step_1d():
    X = np.array([1.0, 2.0, 3.0]).reshape(3, 1)
    median = 2.0
    new_y = _modified_weiszfeld_step(X, median)
    assert_array_almost_equal(new_y, median)
    y = 2.5
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_less(median, new_y)
    assert_array_less(new_y, y)
    y = 3.0
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_less(median, new_y)
    assert_array_less(new_y, y)
    X = np.array([1.0, 2.0, 3.0]).reshape(1, 3)
    y = X[0]
    new_y = _modified_weiszfeld_step(X, y)
    assert_array_equal(y, new_y)