from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_empty_na_values_no_default_with_index(all_parsers):
    data = 'a,1\nb,2'
    parser = all_parsers
    expected = DataFrame({'1': [2]}, index=Index(['b'], name='a'))
    result = parser.read_csv(StringIO(data), index_col=0, keep_default_na=False)
    tm.assert_frame_equal(result, expected)