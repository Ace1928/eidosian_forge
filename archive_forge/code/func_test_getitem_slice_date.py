from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.core.dtypes.common import is_scalar
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexing import IndexingError
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('slc, positions', [[slice(date(2018, 1, 1), None), [0, 1, 2]], [slice(date(2019, 1, 2), None), [2]], [slice(date(2020, 1, 1), None), []], [slice(None, date(2020, 1, 1)), [0, 1, 2]], [slice(None, date(2019, 1, 1)), [0]]])
def test_getitem_slice_date(self, slc, positions):
    ser = Series([0, 1, 2], DatetimeIndex(['2019-01-01', '2019-01-01T06:00:00', '2019-01-02']))
    result = ser[slc]
    expected = ser.take(positions)
    tm.assert_series_equal(result, expected)