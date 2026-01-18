import re
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
def check_replace(to_rep, val, expected):
    sc = ser.copy()
    result = ser.replace(to_rep, val)
    return_value = sc.replace(to_rep, val, inplace=True)
    assert return_value is None
    tm.assert_series_equal(expected, result)
    tm.assert_series_equal(expected, sc)