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
def test_const_indicator():
    rs = np.random.RandomState(12345)
    x = rs.randint(0, 3, size=30)
    x = pd.get_dummies(pd.Series(x, dtype='category'), drop_first=False, dtype=float)
    y = np.dot(x, [1.0, 2.0, 3.0]) + rs.normal(size=30)
    resc = OLS(y, add_constant(x.iloc[:, 1:], prepend=True)).fit()
    res = OLS(y, x, hasconst=True).fit()
    assert_almost_equal(resc.rsquared, res.rsquared, 12)
    assert res.model.data.k_constant == 1
    assert resc.model.data.k_constant == 1