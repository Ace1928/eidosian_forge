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
def test_ljungbox_dof_adj():
    data = sunspots.load_pandas().data['SUNACTIVITY']
    res = AutoReg(data, 4, old_names=False).fit()
    resid = res.resid
    res1 = smsdia.acorr_ljungbox(resid, lags=10)
    res2 = smsdia.acorr_ljungbox(resid, lags=10, model_df=4)
    assert_allclose(res1.iloc[:, 0], res2.iloc[:, 0])
    assert np.all(np.isnan(res2.iloc[:4, 1]))
    assert np.all(res2.iloc[4:, 1] <= res1.iloc[4:, 1])