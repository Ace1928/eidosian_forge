import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_levene(self):
    data = self.data
    statistic = 1.0866123063642
    p_value = 0.3471072204516
    res0 = smo.test_scale_oneway(data, method='equal', center='median', transform='abs', trim_frac_mean=0.2)
    assert_allclose(res0.pvalue, p_value, rtol=1e-13)
    assert_allclose(res0.statistic, statistic, rtol=1e-13)
    statistic = 1.10732113109744
    p_value = 0.340359251994645
    df = [2, 40]
    res0 = smo.test_scale_oneway(data, method='equal', center='trimmed', transform='abs', trim_frac_mean=0.2)
    assert_allclose(res0.pvalue, p_value, rtol=1e-13)
    assert_allclose(res0.statistic, statistic, rtol=1e-13)
    assert_allclose(res0.df, df)
    statistic = 1.07894485177512
    parameter = [2, 40]
    p_value = 0.349641166869223
    res0 = smo.test_scale_oneway(data, method='equal', center='mean', transform='abs', trim_frac_mean=0.2)
    assert_allclose(res0.pvalue, p_value, rtol=1e-13)
    assert_allclose(res0.statistic, statistic, rtol=1e-13)
    assert_allclose(res0.df, parameter)
    statistic = 3.01982414477323
    p_value = 0.220929402900495
    from scipy import stats
    stat, pv = stats.bartlett(*data)
    assert_allclose(pv, p_value, rtol=1e-13)
    assert_allclose(stat, statistic, rtol=1e-13)