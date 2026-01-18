import json
import os
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from statsmodels.datasets import macrodata, sunspots
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.diagnostic as smsdia
import statsmodels.stats.outliers_influence as oi
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
@pytest.mark.smoke
def test_diagnostics_pandas(reset_randomstate):
    n = 100
    df = pd.DataFrame({'y': np.random.rand(n), 'x': np.random.rand(n), 'z': np.random.rand(n)})
    y, x = (df['y'], add_constant(df['x']))
    res = OLS(df['y'], add_constant(df[['x']])).fit()
    res_large = OLS(df['y'], add_constant(df[['x', 'z']])).fit()
    res_other = OLS(df['y'], add_constant(df[['z']])).fit()
    smsdia.linear_reset(res_large)
    smsdia.linear_reset(res_large, test_type='fitted')
    smsdia.linear_reset(res_large, test_type='exog')
    smsdia.linear_reset(res_large, test_type='princomp')
    smsdia.het_goldfeldquandt(y, x)
    smsdia.het_breuschpagan(res.resid, x)
    smsdia.het_white(res.resid, x)
    smsdia.het_arch(res.resid)
    smsdia.acorr_breusch_godfrey(res)
    smsdia.acorr_ljungbox(y)
    smsdia.linear_rainbow(res)
    smsdia.linear_lm(res.resid, x)
    smsdia.linear_harvey_collier(res)
    smsdia.acorr_lm(res.resid)
    smsdia.breaks_cusumolsresid(res.resid)
    smsdia.breaks_hansen(res)
    smsdia.compare_cox(res, res_other)
    smsdia.compare_encompassing(res, res_other)
    smsdia.compare_j(res, res_other)
    smsdia.recursive_olsresiduals(res)
    smsdia.recursive_olsresiduals(res, order_by=np.arange(y.shape[0] - 1, 0 - 1, -1))
    smsdia.spec_white(res.resid, x)