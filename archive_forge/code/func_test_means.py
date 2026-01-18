import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_means(self):
    statistic = 7.10900606421182
    parameter = [2, 31.4207256105052]
    p_value = 0.00283841965791224
    res = anova_oneway(self.data, use_var='bf')
    assert_allclose(res.pvalue2, p_value, rtol=1e-13)
    assert_allclose(res.statistic, statistic, rtol=1e-13)
    assert_allclose([res.df_num2, res.df_denom], parameter)