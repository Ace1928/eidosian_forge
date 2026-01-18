from statsmodels.compat.pandas import (
from datetime import datetime
import numpy as np
from numpy import array, column_stack
from numpy.testing import (
from pandas import DataFrame, concat, date_range
from statsmodels.datasets import macrodata
from statsmodels.tsa.filters._utils import pandas_wrapper
from statsmodels.tsa.filters.bk_filter import bkfilter
from statsmodels.tsa.filters.cf_filter import cffilter
from statsmodels.tsa.filters.filtertools import (
from statsmodels.tsa.filters.hp_filter import hpfilter
def test_convolution2d(self):
    x = self.data.values
    res = convolution_filter(x, [[0.75], [0.25]])
    expected = self.expected.conv2
    np.testing.assert_almost_equal(res, expected[:, None])
    res = convolution_filter(np.c_[x, x], [[0.75, 0.75], [0.25, 0.25]])
    np.testing.assert_almost_equal(res, np.c_[expected, expected])
    res = convolution_filter(x, [[0.75], [0.25]], nsides=1)
    expected = self.expected.conv1
    np.testing.assert_almost_equal(res, expected[:, None])
    x = self.datana.values
    res = convolution_filter(x, [[0.75], [0.25]])
    expected = self.expected.conv2_na
    np.testing.assert_almost_equal(res, expected[:, None])
    res = convolution_filter(x, [[0.75], [0.25]], nsides=1)
    expected = self.expected.conv1_na
    np.testing.assert_almost_equal(res, expected[:, None])