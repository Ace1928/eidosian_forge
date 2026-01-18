from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@skip_pyarrow
@pytest.mark.parametrize('usecols', [['a', 'c'], lambda x: x in ['a', 'c']])
def test_incomplete_first_row(all_parsers, usecols):
    data = '1,2\n1,2,3'
    parser = all_parsers
    names = ['a', 'b', 'c']
    expected = DataFrame({'a': [1, 1], 'c': [np.nan, 3]})
    result = parser.read_csv(StringIO(data), names=names, usecols=usecols)
    tm.assert_frame_equal(result, expected)