from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_detect_string_na(all_parsers):
    parser = all_parsers
    data = 'A,B\nfoo,bar\nNA,baz\nNaN,nan\n'
    expected = DataFrame([['foo', 'bar'], [np.nan, 'baz'], [np.nan, np.nan]], columns=['A', 'B'])
    if parser.engine == 'pyarrow':
        expected.loc[[1, 2], 'A'] = None
        expected.loc[2, 'B'] = None
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)