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
def test_burg_errors():
    with pytest.raises(ValueError):
        burg(np.ones((100, 2)))
    with pytest.raises(ValueError):
        burg(np.random.randn(100), 0)
    with pytest.raises(ValueError):
        burg(np.random.randn(100), 'apple')