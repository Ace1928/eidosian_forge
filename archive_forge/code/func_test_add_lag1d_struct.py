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
def test_add_lag1d_struct(self):
    data = np.zeros(100, dtype=[('variable', float)])
    nddata = self.random_data
    data['variable'] = nddata
    lagmat = stattools.lagmat(nddata, 3, trim='Both', original='in')
    lag_data = tools.add_lag(data, 0, lags=3, insert=True)
    assert_equal(lagmat, lag_data)
    lag_data = tools.add_lag(data, 0, lags=3, insert=False)
    assert_equal(lagmat, lag_data)
    lag_data = tools.add_lag(data, lags=3, insert=True)
    assert_equal(lagmat, lag_data)