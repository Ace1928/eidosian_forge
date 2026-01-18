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
def test_het_breusch_pagan_nonrobust(self):
    res = self.res
    bptest = dict(statistic=1.302014063483341, pvalue=0.5215203247110649, parameters=(2,), distr='f')
    bp = smsdia.het_breuschpagan(res.resid, res.model.exog, robust=False)
    compare_to_reference(bp, bptest, decimal=(12, 12))