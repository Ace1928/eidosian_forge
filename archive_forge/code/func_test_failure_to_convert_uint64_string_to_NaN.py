import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_failure_to_convert_uint64_string_to_NaN():
    result = to_numeric('uint64', errors='coerce')
    assert np.isnan(result)
    ser = Series([32, 64, np.nan])
    result = to_numeric(Series(['32', '64', 'uint64']), errors='coerce')
    tm.assert_series_equal(result, ser)