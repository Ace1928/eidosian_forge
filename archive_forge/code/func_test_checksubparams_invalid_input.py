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
@pytest.mark.parametrize('param, ExceptionCls, match', [({'n_subsamples': 1}, ValueError, re.escape('Invalid parameter since n_features+1 > n_subsamples (2 > 1)')), ({'n_subsamples': 101}, ValueError, re.escape('Invalid parameter since n_subsamples > n_samples (101 > 50)'))])
def test_checksubparams_invalid_input(param, ExceptionCls, match):
    X, y, w, c = gen_toy_problem_1d()
    theil_sen = TheilSenRegressor(**param, random_state=0)
    with pytest.raises(ExceptionCls, match=match):
        theil_sen.fit(X, y)