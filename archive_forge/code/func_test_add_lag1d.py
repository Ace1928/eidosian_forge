from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
def test_add_lag1d(self):
    data = self.random_data
    lagmat = stattools.lagmat(data, 3, trim='Both')
    results = np.column_stack((data[3:], lagmat))
    lag_data = tools.add_lag(data, lags=3, insert=True)
    assert_equal(results, lag_data)
    data = data[:, None]
    lagmat = stattools.lagmat(data, 3, trim='Both')
    results = np.column_stack((data[3:], lagmat))
    lag_data = tools.add_lag(data, lags=3, insert=True)
    assert_equal(results, lag_data)