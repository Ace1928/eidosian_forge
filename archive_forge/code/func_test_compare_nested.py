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
@pytest.mark.parametrize('comp', [smsdia.compare_cox, smsdia.compare_j])
def test_compare_nested(self, comp, diagnostic_data):
    data = diagnostic_data
    res1 = OLS(data.y, data[['c', 'x1']]).fit()
    res2 = OLS(data.y, data[['c', 'x1', 'x2']]).fit()
    with pytest.raises(ValueError, match='The exog in results_x'):
        comp(res1, res2)
    with pytest.raises(ValueError, match='The exog in results_x'):
        comp(res2, res1)