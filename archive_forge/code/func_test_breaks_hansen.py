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
def test_breaks_hansen(self):
    breaks_nyblom_hansen = dict(statistic=1.0300792740544484, pvalue=0.1136087530212015, parameters=(), distr='BB')
    bh = smsdia.breaks_hansen(self.res)
    assert_almost_equal(bh[0], breaks_nyblom_hansen['statistic'], decimal=12)