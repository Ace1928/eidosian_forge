from datetime import (
from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
import pandas.io.formats.format as fmt
def test_datetime64tz_slice_non_truncate(self):
    df = DataFrame({'x': date_range('2019', periods=10, tz='UTC')})
    expected = repr(df)
    df = df.iloc[:, :5]
    result = repr(df)
    assert result == expected