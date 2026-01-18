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
def test_acorr_breusch_godfrey(self):
    res = self.res
    breuschgodfrey_f = dict(statistic=1.179280833676792, pvalue=0.321197487261203, parameters=(4, 195), distr='f')
    breuschgodfrey_c = dict(statistic=4.771042651230007, pvalue=0.3116067133066697, parameters=(4,), distr='chi2')
    bg = smsdia.acorr_breusch_godfrey(res, nlags=4)
    bg_r = [breuschgodfrey_c['statistic'], breuschgodfrey_c['pvalue'], breuschgodfrey_f['statistic'], breuschgodfrey_f['pvalue']]
    assert_almost_equal(bg, bg_r, decimal=11)
    bg2 = smsdia.acorr_breusch_godfrey(res, nlags=None)
    bg3 = smsdia.acorr_breusch_godfrey(res, nlags=10)
    assert_almost_equal(bg2, bg3, decimal=12)