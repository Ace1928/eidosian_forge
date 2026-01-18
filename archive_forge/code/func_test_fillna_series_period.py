from __future__ import annotations
from datetime import (
import itertools
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas as pd
import pandas._testing as tm
@pytest.mark.parametrize('fill_val', [1, 1.1, 1 + 1j, True, pd.Interval(1, 2, closed='left'), pd.Timestamp('2012-01-01', tz='US/Eastern'), pd.Timestamp('2012-01-01'), pd.Timedelta(days=1), pd.Period('2016-01-01', 'W')])
def test_fillna_series_period(self, index_or_series, fill_val):
    pi = pd.period_range('2016-01-01', periods=4, freq='D').insert(1, pd.NaT)
    assert isinstance(pi.dtype, pd.PeriodDtype)
    obj = index_or_series(pi)
    exp = index_or_series([pi[0], fill_val, pi[2], pi[3], pi[4]], dtype=object)
    fill_dtype = object
    self._assert_fillna_conversion(obj, fill_val, exp, fill_dtype)