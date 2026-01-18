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
def test_lagmat2ds_pandas(self):
    data = self.macro_df
    lagmat = stattools.lagmat2ds(data, 2)
    expected = self._prepare_expected(data.values, 2)
    assert_array_equal(lagmat, expected)
    lagmat = stattools.lagmat2ds(data.iloc[:, :2], 3, trim='both')
    expected = self._prepare_expected(data.values[:, :2], 3)
    expected = expected[3:]
    assert_array_equal(lagmat, expected)
    data = self.series
    lagmat = stattools.lagmat2ds(data, 5)
    expected = self._prepare_expected(data.values[:, None], 5)
    assert_array_equal(lagmat, expected)