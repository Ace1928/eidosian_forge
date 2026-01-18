from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
def test_get_indexer(self):
    idx = date_range('2000-01-01', periods=3)
    exp = np.array([0, 1, 2], dtype=np.intp)
    tm.assert_numpy_array_equal(idx.get_indexer(idx), exp)
    target = idx[0] + pd.to_timedelta(['-1 hour', '12 hours', '1 day 1 hour'])
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'pad'), np.array([-1, 0, 1], dtype=np.intp))
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'backfill'), np.array([0, 1, 2], dtype=np.intp))
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest'), np.array([0, 1, 1], dtype=np.intp))
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest', tolerance=pd.Timedelta('1 hour')), np.array([0, -1, 1], dtype=np.intp))
    tol_raw = [pd.Timedelta('1 hour'), pd.Timedelta('1 hour'), pd.Timedelta('1 hour').to_timedelta64()]
    tm.assert_numpy_array_equal(idx.get_indexer(target, 'nearest', tolerance=[np.timedelta64(x) for x in tol_raw]), np.array([0, -1, 1], dtype=np.intp))
    tol_bad = [pd.Timedelta('2 hour').to_timedelta64(), pd.Timedelta('1 hour').to_timedelta64(), 'foo']
    msg = "Could not convert 'foo' to NumPy timedelta"
    with pytest.raises(ValueError, match=msg):
        idx.get_indexer(target, 'nearest', tolerance=tol_bad)
    with pytest.raises(ValueError, match='abbreviation w/o a number'):
        idx.get_indexer(idx[[0]], method='nearest', tolerance='foo')