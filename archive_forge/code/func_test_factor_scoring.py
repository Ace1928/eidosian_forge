import warnings
from statsmodels.compat.pandas import PD_LT_1_4
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.factor import Factor
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
@pytest.mark.smoke
def test_factor_scoring():
    path = os.path.abspath(__file__)
    dir_path = os.path.dirname(path)
    csv_path = os.path.join(dir_path, 'results', 'factor_data.csv')
    y = pd.read_csv(csv_path)
    csv_path = os.path.join(dir_path, 'results', 'factors_stata.csv')
    f_s = pd.read_csv(csv_path)
    mod = Factor(y, 2)
    res = mod.fit(maxiter=1)
    res.rotate('varimax')
    f_reg = res.factor_scoring(method='reg')
    assert_allclose(f_reg * [1, -1], f_s[['f1', 'f2']].values, atol=0.0001, rtol=0.001)
    f_bart = res.factor_scoring()
    assert_allclose(f_bart * [1, -1], f_s[['f1b', 'f2b']].values, atol=0.0001, rtol=0.001)
    f_ols = res.factor_scoring(method='ols')
    f_gls = res.factor_scoring(method='gls')
    f_reg_z = _zscore(f_reg)
    f_ols_z = _zscore(f_ols)
    f_gls_z = _zscore(f_gls)
    assert_array_less(0.98, (f_ols_z * f_reg_z).mean(0))
    assert_array_less(0.999, (f_gls_z * f_reg_z).mean(0))
    res.rotate('oblimin')
    assert_allclose(res._corr_factors()[0, 1], -1 * 0.25651037, rtol=0.001)
    f_reg = res.factor_scoring(method='reg')
    assert_allclose(f_reg * [1, -1], f_s[['f1o', 'f2o']].values, atol=0.0001, rtol=0.001)
    f_bart = res.factor_scoring()
    assert_allclose(f_bart * [1, -1], f_s[['f1ob', 'f2ob']].values, atol=0.0001, rtol=0.001)
    f_ols = res.factor_scoring(method='ols')
    f_gls = res.factor_scoring(method='gls')
    f_reg_z = _zscore(f_reg)
    f_ols_z = _zscore(f_ols)
    f_gls_z = _zscore(f_gls)
    assert_array_less(0.97, (f_ols_z * f_reg_z).mean(0))
    assert_array_less(0.999, (f_gls_z * f_reg_z).mean(0))
    f_ols2 = res.factor_scoring(method='ols', endog=res.model.endog)
    assert_allclose(f_ols2, f_ols, rtol=1e-13)