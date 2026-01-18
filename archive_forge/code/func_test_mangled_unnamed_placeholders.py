from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@xfail_pyarrow
def test_mangled_unnamed_placeholders(all_parsers):
    orig_key = '0'
    parser = all_parsers
    orig_value = [1, 2, 3]
    df = DataFrame({orig_key: orig_value})
    for i in range(3):
        expected = DataFrame()
        for j in range(i + 1):
            col_name = 'Unnamed: 0' + f'.{1 * j}' * min(j, 1)
            expected.insert(loc=0, column=col_name, value=[0, 1, 2])
        expected[orig_key] = orig_value
        df = parser.read_csv(StringIO(df.to_csv()))
        tm.assert_frame_equal(df, expected)