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
def test_cusum_ols(self):
    cusum_ols = dict(statistic=1.055750610401214, pvalue=0.2149567397376543, parameters=(), distr='BB')
    k_vars = 3
    cs_ols = smsdia.breaks_cusumolsresid(self.res.resid, ddof=k_vars)
    compare_to_reference(cs_ols, cusum_ols, decimal=(12, 12))