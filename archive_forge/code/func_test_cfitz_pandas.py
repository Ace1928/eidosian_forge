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
def test_cfitz_pandas():
    dta = macrodata.load_pandas().data
    index = date_range(start='1959-01-01', end='2009-10-01', freq=QUARTER_END)
    dta.index = index
    cycle, trend = cffilter(dta['infl'])
    ndcycle, ndtrend = cffilter(dta['infl'].values)
    assert_allclose(cycle.values, ndcycle, rtol=1e-14)
    assert_equal(cycle.index[0], datetime(1959, 3, 31))
    assert_equal(cycle.index[-1], datetime(2009, 9, 30))
    assert_equal(cycle.name, 'infl_cycle')
    cycle, trend = cffilter(dta[['infl', 'unemp']])
    ndcycle, ndtrend = cffilter(dta[['infl', 'unemp']].values)
    assert_allclose(cycle.values, ndcycle, rtol=1e-14)
    assert_equal(cycle.index[0], datetime(1959, 3, 31))
    assert_equal(cycle.index[-1], datetime(2009, 9, 30))
    assert_equal(cycle.columns.values, ['infl_cycle', 'unemp_cycle'])