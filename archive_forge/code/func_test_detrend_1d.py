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
def test_detrend_1d(self):
    data = self.data_1d
    assert_array_almost_equal(tools.detrend(data, order=1), np.zeros_like(data))
    assert_array_almost_equal(tools.detrend(data, order=0), [-2, -1, 0, 1, 2])