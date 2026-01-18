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
def gen_toy_problem_2d():
    random_state = np.random.RandomState(0)
    n_samples = 100
    X = random_state.normal(size=(n_samples, 2))
    w = np.array([5.0, 10.0])
    c = 1.0
    noise = 0.1 * random_state.normal(size=n_samples)
    y = np.dot(X, w) + c + noise
    n_outliers = n_samples // 10
    ix = random_state.randint(0, n_samples, size=n_outliers)
    y[ix] = 50 * random_state.normal(size=n_outliers)
    return (X, y, w, c)