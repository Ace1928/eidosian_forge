from io import StringIO
import pytest
from pandas import DataFrame
import pandas._testing as tm
@xfail_pyarrow
def test_mangle_dupe_cols_already_exists_unnamed_col(all_parsers):
    parser = all_parsers
    data = ',Unnamed: 0,,Unnamed: 2\n1,2,3,4'
    result = parser.read_csv(StringIO(data))
    expected = DataFrame([[1, 2, 3, 4]], columns=['Unnamed: 0.1', 'Unnamed: 0', 'Unnamed: 2.1', 'Unnamed: 2'])
    tm.assert_frame_equal(result, expected)