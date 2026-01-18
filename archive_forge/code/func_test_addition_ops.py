from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_addition_ops(self):
    tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
    dti = pd.date_range('20130101', periods=3, name='bar')
    td = Timedelta('1 days')
    dt = Timestamp('20130101')
    result = tdi + dt
    expected = DatetimeIndex(['20130102', NaT, '20130103'], dtype='M8[ns]', name='foo')
    tm.assert_index_equal(result, expected)
    result = dt + tdi
    expected = DatetimeIndex(['20130102', NaT, '20130103'], dtype='M8[ns]', name='foo')
    tm.assert_index_equal(result, expected)
    result = td + tdi
    expected = TimedeltaIndex(['2 days', NaT, '3 days'], name='foo')
    tm.assert_index_equal(result, expected)
    result = tdi + td
    expected = TimedeltaIndex(['2 days', NaT, '3 days'], name='foo')
    tm.assert_index_equal(result, expected)
    msg = 'cannot add indices of unequal length'
    with pytest.raises(ValueError, match=msg):
        tdi + dti[0:1]
    with pytest.raises(ValueError, match=msg):
        tdi[0:1] + dti
    msg = 'Addition/subtraction of integers and integer-arrays'
    with pytest.raises(TypeError, match=msg):
        tdi + Index([1, 2, 3], dtype=np.int64)
    result = tdi + dti
    expected = DatetimeIndex(['20130102', NaT, '20130105'], dtype='M8[ns]')
    tm.assert_index_equal(result, expected)
    result = dti + tdi
    expected = DatetimeIndex(['20130102', NaT, '20130105'], dtype='M8[ns]')
    tm.assert_index_equal(result, expected)
    result = dt + td
    expected = Timestamp('20130102')
    assert result == expected
    result = td + dt
    expected = Timestamp('20130102')
    assert result == expected