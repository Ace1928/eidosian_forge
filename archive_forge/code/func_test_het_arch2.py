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
def test_het_arch2(self):
    resid = self.res.resid
    res1 = smsdia.het_arch(resid, nlags=5, store=True)
    rs1 = res1[-1]
    res2 = smsdia.het_arch(resid, nlags=5, store=True)
    rs2 = res2[-1]
    assert_almost_equal(rs2.resols.params, rs1.resols.params, decimal=12)
    assert_almost_equal(res2[:4], res1[:4], decimal=12)
    res3 = smsdia.het_arch(resid, nlags=5)
    assert_almost_equal(res3[:4], res1[:4], decimal=12)