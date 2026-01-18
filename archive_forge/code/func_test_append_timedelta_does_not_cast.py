from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p24
from pandas.errors import IndexingError
from pandas.core.dtypes.common import is_list_like
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('td', [Timedelta('9 days'), Timedelta('9 days').to_timedelta64(), Timedelta('9 days').to_pytimedelta()])
def test_append_timedelta_does_not_cast(self, td, using_infer_string, request):
    if using_infer_string and (not isinstance(td, Timedelta)):
        request.applymarker(pytest.mark.xfail(reason='inferred as string'))
    expected = Series(['x', td], index=[0, 'td'], dtype=object)
    ser = Series(['x'])
    ser['td'] = td
    tm.assert_series_equal(ser, expected)
    assert isinstance(ser['td'], Timedelta)
    ser = Series(['x'])
    ser.loc['td'] = Timedelta('9 days')
    tm.assert_series_equal(ser, expected)
    assert isinstance(ser['td'], Timedelta)