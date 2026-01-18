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
def test_het_breusch_pagan_1d_err(self):
    res = self.res
    x = np.asarray(res.model.exog)[:, -1]
    with pytest.raises(ValueError, match='The Breusch-Pagan'):
        smsdia.het_breuschpagan(res.resid, x)
    x = np.ones_like(x)
    with pytest.raises(ValueError, match='The Breusch-Pagan'):
        smsdia.het_breuschpagan(res.resid, x)
    x = np.asarray(res.model.exog).copy()
    x[:, 0] = 0
    with pytest.raises(ValueError, match='The Breusch-Pagan'):
        smsdia.het_breuschpagan(res.resid, x)