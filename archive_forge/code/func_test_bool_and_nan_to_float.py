from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_bool_and_nan_to_float(all_parsers):
    parser = all_parsers
    data = '0\nNaN\nTrue\nFalse\n'
    result = parser.read_csv(StringIO(data), dtype='float')
    expected = DataFrame.from_dict({'0': [np.nan, 1.0, 0.0]})
    tm.assert_frame_equal(result, expected)