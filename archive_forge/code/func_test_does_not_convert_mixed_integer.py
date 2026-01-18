from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_does_not_convert_mixed_integer(self):
    df = DataFrame(np.ones((3, 2)), columns=date_range('2020-01-01', periods=2))
    cols = df.columns.join(df.index, how='outer')
    joined = cols.join(df.columns)
    assert cols.dtype == np.dtype('O')
    assert cols.dtype == joined.dtype
    tm.assert_numpy_array_equal(cols.values, joined.values)