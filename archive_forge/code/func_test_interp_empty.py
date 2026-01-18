import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_empty(self):
    df = DataFrame()
    result = df.interpolate()
    assert result is not df
    expected = df
    tm.assert_frame_equal(result, expected)