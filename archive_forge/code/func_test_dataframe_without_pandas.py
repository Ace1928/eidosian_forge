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
def test_dataframe_without_pandas(self):
    data = self.macro_df
    both = stattools.lagmat(data, 3, trim='both', original='in')
    both_np = stattools.lagmat(data.values, 3, trim='both', original='in')
    assert_equal(both, both_np)
    lags = stattools.lagmat(data, 3, trim='none', original='ex')
    lags_np = stattools.lagmat(data.values, 3, trim='none', original='ex')
    assert_equal(lags, lags_np)
    lags, lead = stattools.lagmat(data, 3, trim='forward', original='sep')
    lags_np, lead_np = stattools.lagmat(data.values, 3, trim='forward', original='sep')
    assert_equal(lags, lags_np)
    assert_equal(lead, lead_np)