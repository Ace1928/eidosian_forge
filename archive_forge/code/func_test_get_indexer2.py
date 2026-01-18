from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs.tslibs import period as libperiod
from pandas.errors import InvalidIndexError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_indexer2(self):
    idx = period_range('2000-01-01', periods=3).asfreq('h', how='start')
    tm.assert_numpy_array_equal(idx.get_indexer(idx), np.array([0, 1, 2], dtype=np.intp))
    target = PeriodIndex(['1999-12-31T23', '2000-01-01T12', '2000-01-02T01'], freq='h')
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'pad'), np.array([-1, 0, 1], dtype=np.intp))
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'backfill'), np.array([0, 1, 2], dtype=np.intp))
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest'), np.array([0, 1, 1], dtype=np.intp))
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest', tolerance='1 hour'), np.array([0, -1, 1], dtype=np.intp))
    msg = 'Input has different freq=None from PeriodArray\\(freq=h\\)'
    with pytest.raises(ValueError, match=msg):
        idx.get_indexer(target, 'nearest', tolerance='1 minute')
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest', tolerance='1 day'), np.array([0, 1, 1], dtype=np.intp))
    tol_raw = [Timedelta('1 hour'), Timedelta('1 hour'), np.timedelta64(1, 'D')]
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest', tolerance=[np.timedelta64(x) for x in tol_raw]), np.array([0, -1, 1], dtype=np.intp))
    tol_bad = [Timedelta('2 hour').to_timedelta64(), Timedelta('1 hour').to_timedelta64(), np.timedelta64(1, 'M')]
    with pytest.raises(libperiod.IncompatibleFrequency, match='Input has different freq=None from'):
        idx.get_indexer(target, 'nearest', tolerance=tol_bad)