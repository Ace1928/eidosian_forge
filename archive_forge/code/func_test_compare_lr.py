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
def test_compare_lr(self):
    res = self.res
    res3 = self.res3
    lrtest = dict(loglike1=-763.9752181602237, loglike2=-766.3091902020184, chi2value=4.66794408358942, pvalue=0.03073069384028677, df=(4, 3, 1))
    lrt = res.compare_lr_test(res3)
    assert_almost_equal(lrt[0], lrtest['chi2value'], decimal=11)
    assert_almost_equal(lrt[1], lrtest['pvalue'], decimal=11)
    waldtest = dict(fvalue=4.65216373312492, pvalue=0.03221346195239025, df=(199, 200, 1))
    wt = res.compare_f_test(res3)
    assert_almost_equal(wt[0], waldtest['fvalue'], decimal=11)
    assert_almost_equal(wt[1], waldtest['pvalue'], decimal=11)