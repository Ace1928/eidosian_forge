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
def gen_toy_problem_1d(intercept=True):
    random_state = np.random.RandomState(0)
    w = 3.0
    if intercept:
        c = 2.0
        n_samples = 50
    else:
        c = 0.1
        n_samples = 100
    x = random_state.normal(size=n_samples)
    noise = 0.1 * random_state.normal(size=n_samples)
    y = w * x + c + noise
    if intercept:
        x[42], y[42] = (-2, 4)
        x[43], y[43] = (-2.5, 8)
        x[33], y[33] = (2.5, 1)
        x[49], y[49] = (2.1, 2)
    else:
        x[42], y[42] = (-2, 4)
        x[43], y[43] = (-2.5, 8)
        x[53], y[53] = (2.5, 1)
        x[60], y[60] = (2.1, 2)
        x[72], y[72] = (1.8, -7)
    return (x[:, np.newaxis], y, w, c)