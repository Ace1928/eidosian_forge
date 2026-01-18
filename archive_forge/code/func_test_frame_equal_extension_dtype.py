import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
def test_frame_equal_extension_dtype(frame_or_series, any_numeric_ea_dtype):
    obj = frame_or_series([1, 2], dtype=any_numeric_ea_dtype)
    tm.assert_equal(obj, obj, check_exact=True)