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
def test_ljungbox_errors_warnings():
    data = sunspots.load_pandas().data['SUNACTIVITY']
    with pytest.raises(ValueError, match='model_df must'):
        smsdia.acorr_ljungbox(data, model_df=-1)
    with pytest.raises(ValueError, match='period must'):
        smsdia.acorr_ljungbox(data, model_df=-1, period=1)
    with pytest.raises(ValueError, match='period must'):
        smsdia.acorr_ljungbox(data, model_df=-1, period=-2)
    smsdia.acorr_ljungbox(data)
    ret = smsdia.acorr_ljungbox(data, lags=10)
    assert isinstance(ret, pd.DataFrame)