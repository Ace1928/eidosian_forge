import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_astype_conversion(self):
    idx = PeriodIndex(['2016-05-16', 'NaT', NaT, np.nan], freq='D', name='idx')
    result = idx.astype(object)
    expected = Index([Period('2016-05-16', freq='D')] + [Period(NaT, freq='D')] * 3, dtype='object', name='idx')
    tm.assert_index_equal(result, expected)
    result = idx.astype(np.int64)
    expected = Index([16937] + [-9223372036854775808] * 3, dtype=np.int64, name='idx')
    tm.assert_index_equal(result, expected)
    result = idx.astype(str)
    expected = Index([str(x) for x in idx], name='idx', dtype=object)
    tm.assert_index_equal(result, expected)
    idx = period_range('1990', '2009', freq='Y', name='idx')
    result = idx.astype('i8')
    tm.assert_index_equal(result, Index(idx.asi8, name='idx'))
    tm.assert_numpy_array_equal(result.values, idx.asi8)