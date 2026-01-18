from datetime import datetime
from io import StringIO
import os
import pytest
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('data,expected,header', [('a,b', DataFrame(columns=['a', 'b']), [0]), ('a,b\nc,d', DataFrame(columns=MultiIndex.from_tuples([('a', 'c'), ('b', 'd')])), [0, 1])])
@pytest.mark.parametrize('round_trip', [True, False])
def test_multi_index_blank_df(all_parsers, data, expected, header, round_trip):
    parser = all_parsers
    data = expected.to_csv(index=False) if round_trip else data
    result = parser.read_csv(StringIO(data), header=header)
    tm.assert_frame_equal(result, expected)