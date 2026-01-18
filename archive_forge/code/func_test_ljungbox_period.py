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
def test_ljungbox_period():
    data = sunspots.load_pandas().data['SUNACTIVITY']
    ar_res = AutoReg(data, 4, old_names=False).fit()
    res = smsdia.acorr_ljungbox(ar_res.resid, period=13)
    res2 = smsdia.acorr_ljungbox(ar_res.resid, lags=26)
    assert_frame_equal(res, res2)