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
def test_acorr_ljung_box_small_default(self):
    res = self.res
    ljung_box_small = dict(statistic=9.61503968281915, pvalue=0.72507000996945, parameters=(0,), distr='chi2')
    ljung_box_bp_small = dict(statistic=7.41692150864936, pvalue=0.87940785887006, parameters=(0,), distr='chi2')
    if isinstance(res.resid, np.ndarray):
        resid = res.resid[:30]
    else:
        resid = res.resid.iloc[:30]
    df = smsdia.acorr_ljungbox(resid, boxpierce=True, lags=13)
    idx = df.index.max()
    compare_to_reference([df.loc[idx, 'lb_stat'], df.loc[idx, 'lb_pvalue']], ljung_box_small, decimal=(12, 12))
    compare_to_reference([df.loc[idx, 'bp_stat'], df.loc[idx, 'bp_pvalue']], ljung_box_bp_small, decimal=(12, 12))