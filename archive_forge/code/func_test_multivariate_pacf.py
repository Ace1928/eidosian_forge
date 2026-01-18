import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_equal, assert_array_less,
import pandas as pd
from scipy.linalg import solve_discrete_lyapunov
from statsmodels.tsa.statespace import tools
from statsmodels.tsa.stattools import acovf
def test_multivariate_pacf():
    np.random.seed(1234)
    x = np.arange(10000)
    y = np.random.normal(size=10000)
    assert_allclose(tools._compute_multivariate_sample_pacf(np.c_[x, y], maxlag=1)[0], np.diag([1, 0]), atol=0.01)