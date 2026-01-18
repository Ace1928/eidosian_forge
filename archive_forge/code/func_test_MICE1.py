import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def test_MICE1(self):
    df = gendat()
    imp_data = mice.MICEData(df)
    mi = mice.MICE('y ~ x1 + x2 + x1:x2', sm.OLS, imp_data)
    from statsmodels.regression.linear_model import RegressionResultsWrapper
    for j in range(3):
        x = mi.next_sample()
        assert issubclass(x.__class__, RegressionResultsWrapper)