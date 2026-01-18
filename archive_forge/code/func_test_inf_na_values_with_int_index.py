from io import StringIO
import numpy as np
import pytest
from pandas._libs.parsers import STR_NA_VALUES
from pandas import (
import pandas._testing as tm
def test_inf_na_values_with_int_index(all_parsers):
    parser = all_parsers
    data = 'idx,col1,col2\n1,3,4\n2,inf,-inf'
    out = parser.read_csv(StringIO(data), index_col=[0], na_values=['inf', '-inf'])
    expected = DataFrame({'col1': [3, np.nan], 'col2': [4, np.nan]}, index=Index([1, 2], name='idx'))
    tm.assert_frame_equal(out, expected)