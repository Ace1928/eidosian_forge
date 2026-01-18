from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@xfail_pyarrow
def test_basic_names(all_parsers):
    parser = all_parsers
    data = 'a,b,a\n0,1,2\n3,4,5'
    expected = DataFrame([[0, 1, 2], [3, 4, 5]], columns=['a', 'b', 'a.1'])
    result = parser.read_csv(StringIO(data))
    tm.assert_frame_equal(result, expected)