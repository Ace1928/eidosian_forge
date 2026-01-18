import os
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
import pandas as pd
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import mlemodel, sarimax, varmax
from statsmodels.tsa.statespace.tests.test_impulse_responses import TVSS
from statsmodels.tsa.statespace.kalman_filter import FILTER_UNIVARIATE
from statsmodels.tsa.statespace.kalman_smoother import (
def get_acov_model(missing, filter_univariate, tvp, oos=None, params=None, return_ssm=True):
    dta = datasets.macrodata.load_pandas().data
    dta.index = pd.date_range(start='1959-01-01', end='2009-7-01', freq='QS')
    endog = np.log(dta[['realgdp', 'realcons']]).diff().iloc[1:]
    if missing == 'all':
        endog.iloc[:5, :] = np.nan
        endog.iloc[11:13, :] = np.nan
    elif missing == 'partial':
        endog.iloc[0:5, 0] = np.nan
        endog.iloc[11:13, 0] = np.nan
    elif missing == 'mixed':
        endog.iloc[0:5, 0] = np.nan
        endog.iloc[1:7, 1] = np.nan
        endog.iloc[11:13, 0] = np.nan
    if oos is not None:
        new_ix = pd.date_range(start=endog.index[0], periods=len(endog) + oos, freq='QS')
        endog = endog.reindex(new_ix)
    if not tvp:
        mod = varmax.VARMAX(endog, order=(4, 0, 0), measurement_error=True, tolerance=0)
        mod.ssm.filter_univariate = filter_univariate
        if params is None:
            params = mod.start_params
        res = mod.smooth(params, return_ssm=return_ssm)
    else:
        mod = TVSSWithLags(endog)
        mod.ssm.filter_univariate = filter_univariate
        res = mod.smooth([], return_ssm=return_ssm)
    return (mod, res)