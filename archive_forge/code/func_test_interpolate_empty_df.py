import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interpolate_empty_df(self):
    df = DataFrame()
    expected = df.copy()
    result = df.interpolate(inplace=True)
    assert result is None
    tm.assert_frame_equal(df, expected)