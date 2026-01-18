from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('kwargs', [{'header': [0, 1]}, {'skiprows': 3, 'names': [('a', 'q'), ('a', 'r'), ('a', 's'), ('b', 't'), ('c', 'u'), ('c', 'v')]}, {'skiprows': 3, 'names': [_TestTuple('a', 'q'), _TestTuple('a', 'r'), _TestTuple('a', 's'), _TestTuple('b', 't'), _TestTuple('c', 'u'), _TestTuple('c', 'v')]}])
def test_header_multi_index_common_format1(all_parsers, kwargs):
    parser = all_parsers
    expected = DataFrame([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], index=['one', 'two'], columns=MultiIndex.from_tuples([('a', 'q'), ('a', 'r'), ('a', 's'), ('b', 't'), ('c', 'u'), ('c', 'v')]))
    data = ',a,a,a,b,c,c\n,q,r,s,t,u,v\n,,,,,,\none,1,2,3,4,5,6\ntwo,7,8,9,10,11,12'
    result = parser.read_csv(StringIO(data), index_col=0, **kwargs)
    tm.assert_frame_equal(result, expected)