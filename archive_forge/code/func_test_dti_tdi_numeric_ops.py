from datetime import (
import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import NumpyExtensionArray
from pandas.tests.arithmetic.common import (
def test_dti_tdi_numeric_ops(self):
    tdi = TimedeltaIndex(['1 days', NaT, '2 days'], name='foo')
    dti = pd.date_range('20130101', periods=3, name='bar')
    result = tdi - tdi
    expected = TimedeltaIndex(['0 days', NaT, '0 days'], name='foo')
    tm.assert_index_equal(result, expected)
    result = tdi + tdi
    expected = TimedeltaIndex(['2 days', NaT, '4 days'], name='foo')
    tm.assert_index_equal(result, expected)
    result = dti - tdi
    expected = DatetimeIndex(['20121231', NaT, '20130101'], dtype='M8[ns]')
    tm.assert_index_equal(result, expected)