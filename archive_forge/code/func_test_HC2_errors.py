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
def test_HC2_errors(self):
    assert_almost_equal(self.res1.HC2_se[:-1], self.res2.HC2_se[:-1], DECIMAL_4)
    assert_allclose(self.res1.HC2_se[-1], self.res2.HC2_se[-1], rtol=5e-07)