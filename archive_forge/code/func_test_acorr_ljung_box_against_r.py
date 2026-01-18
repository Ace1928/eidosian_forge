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
def test_acorr_ljung_box_against_r(self, reset_randomstate):
    rs = np.random.RandomState(9876543)
    y1 = rs.standard_normal(100)
    e = rs.standard_normal(201)
    y2 = np.zeros_like(e)
    y2[0] = e[0]
    for i in range(1, 201):
        y2[i] = 0.5 * y2[i - 1] - 0.4 * e[i - 1] + e[i]
    y2 = y2[-100:]
    r_results_y1_lb = [[0.15685, 1, 0.6921], [5.4737, 5, 0.3608], [10.508, 10, 0.3971]]
    r_results_y2_lb = [[2.8764, 1, 0.08989], [3.8104, 5, 0.577], [8.4779, 10, 0.5823]]
    res_y1 = smsdia.acorr_ljungbox(y1, 10)
    res_y2 = smsdia.acorr_ljungbox(y2, 10)
    for i, loc in enumerate((1, 5, 10)):
        row = res_y1.loc[loc]
        assert_allclose(r_results_y1_lb[i][0], row.loc['lb_stat'], rtol=0.001)
        assert_allclose(r_results_y1_lb[i][2], row.loc['lb_pvalue'], rtol=0.001)
        row = res_y2.loc[loc]
        assert_allclose(r_results_y2_lb[i][0], row.loc['lb_stat'], rtol=0.001)
        assert_allclose(r_results_y2_lb[i][2], row.loc['lb_pvalue'], rtol=0.001)
    res = smsdia.acorr_ljungbox(y2, 10, boxpierce=True)
    assert_allclose(res.loc[10, 'bp_stat'], 7.8935, rtol=0.001)
    assert_allclose(res.loc[10, 'bp_pvalue'], 0.639, rtol=0.001)
    res = smsdia.acorr_ljungbox(y2, 10, boxpierce=True, model_df=1)
    assert_allclose(res.loc[10, 'bp_pvalue'], 0.5449, rtol=0.001)