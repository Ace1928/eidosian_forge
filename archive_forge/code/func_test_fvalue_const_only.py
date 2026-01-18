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
def test_fvalue_const_only():
    rs = np.random.RandomState(12345)
    x = rs.randint(0, 3, size=30)
    x = pd.get_dummies(pd.Series(x, dtype='category'), drop_first=False, dtype=float)
    x[x.columns[0]] = 1
    y = np.dot(x, [1.0, 2.0, 3.0]) + rs.normal(size=30)
    res = OLS(y, x, hasconst=True).fit(cov_type='HC1')
    assert not np.isnan(res.fvalue)
    assert isinstance(res.fvalue, float)
    assert isinstance(res.f_pvalue, float)