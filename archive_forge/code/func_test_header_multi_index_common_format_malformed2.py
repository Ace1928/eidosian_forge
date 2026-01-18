from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
def test_header_multi_index_common_format_malformed2(all_parsers):
    parser = all_parsers
    expected = DataFrame(np.array([[2, 3, 4, 5, 6], [8, 9, 10, 11, 12]], dtype='int64'), index=Index([1, 7]), columns=MultiIndex(levels=[['a', 'b', 'c'], ['r', 's', 't', 'u', 'v']], codes=[[0, 0, 1, 2, 2], [0, 1, 2, 3, 4]], names=[None, 'q']))
    data = ',a,a,b,c,c\nq,r,s,t,u,v\n1,2,3,4,5,6\n7,8,9,10,11,12'
    result = parser.read_csv(StringIO(data), header=[0, 1], index_col=0)
    tm.assert_frame_equal(expected, result)