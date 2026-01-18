import numpy as np
import pytest
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('tz', [None, 'US/Central'])
def test_asarray_object_dt64(self, tz):
    ser = Series(date_range('2000', periods=2, tz=tz))
    with tm.assert_produces_warning(None):
        result = np.asarray(ser, dtype=object)
    expected = np.array([Timestamp('2000-01-01', tz=tz), Timestamp('2000-01-02', tz=tz)])
    tm.assert_numpy_array_equal(result, expected)