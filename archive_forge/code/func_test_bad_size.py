from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_bad_size():
    np.random.seed(54321)
    data = np.random.uniform(0, 20, 31)
    assert_raises(ValueError, OLS, data, data[1:])