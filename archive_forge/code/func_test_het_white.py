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
def test_het_white(self):
    res = self.res
    hw = smsdia.het_white(res.resid, res.model.exog)
    hw_values = (33.50372289653844, 2.988796059783026e-06, 7.794510122843095, 1.0354575277704231e-06)
    assert_almost_equal(hw, hw_values)