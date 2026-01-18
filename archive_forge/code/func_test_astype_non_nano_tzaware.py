from __future__ import annotations
from datetime import timedelta
import operator
import numpy as np
import pytest
from pandas._libs.tslibs import tz_compare
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import (
def test_astype_non_nano_tzaware(self):
    dti = pd.date_range('2016-01-01', periods=3, tz='UTC')
    res = dti.astype('M8[s, US/Pacific]')
    assert res.dtype == 'M8[s, US/Pacific]'
    dta = dti._data
    res = dta.astype('M8[s, US/Pacific]')
    assert res.dtype == 'M8[s, US/Pacific]'
    res2 = res.astype('M8[s, UTC]')
    assert res2.dtype == 'M8[s, UTC]'
    assert not tm.shares_memory(res2, res)
    res3 = res.astype('M8[s, UTC]', copy=False)
    assert res2.dtype == 'M8[s, UTC]'
    assert tm.shares_memory(res3, res)