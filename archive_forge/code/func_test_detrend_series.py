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
def test_detrend_series(self):
    data = pd.Series(self.data_1d, name='one')
    detrended = tools.detrend(data, order=1)
    assert_array_almost_equal(detrended.values, np.zeros_like(data))
    assert_series_equal(detrended, pd.Series(detrended.values, name='one'))
    detrended = tools.detrend(data, order=0)
    assert_array_almost_equal(detrended.values, pd.Series([-2, -1, 0, 1, 2]))
    assert_series_equal(detrended, pd.Series(detrended.values, name='one'))