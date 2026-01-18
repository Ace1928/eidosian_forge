from io import StringIO
import pytest
from pandas import (
import pandas._testing as tm
def test_iterator2(all_parsers):
    parser = all_parsers
    data = 'A,B,C\nfoo,1,2,3\nbar,4,5,6\nbaz,7,8,9\n'
    if parser.engine == 'pyarrow':
        msg = "The 'iterator' option is not supported with the 'pyarrow' engine"
        with pytest.raises(ValueError, match=msg):
            parser.read_csv(StringIO(data), iterator=True)
        return
    with parser.read_csv(StringIO(data), iterator=True) as reader:
        result = list(reader)
    expected = DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['foo', 'bar', 'baz'], columns=['A', 'B', 'C'])
    tm.assert_frame_equal(result[0], expected)