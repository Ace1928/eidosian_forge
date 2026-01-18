from io import (
import numpy as np
import pytest
import pandas._libs.parsers as parser
from pandas._libs.parsers import TextReader
from pandas.errors import ParserWarning
from pandas import DataFrame
import pandas._testing as tm
from pandas.io.parsers import (
from pandas.io.parsers.c_parser_wrapper import ensure_dtype_objs
@pytest.mark.parametrize('repeat', range(10))
def test_empty_field_eof_mem_access_bug(self, repeat):
    a = DataFrame([['b'], [np.nan]], columns=['a'], index=['a', 'c'])
    b = DataFrame([[1, 1, 1, 0], [1, 1, 1, 0]], columns=list('abcd'), index=[1, 1])
    c = DataFrame([[1, 2, 3, 4], [6, np.nan, np.nan, np.nan], [8, 9, 10, 11], [13, 14, np.nan, np.nan]], columns=list('abcd'), index=[0, 5, 7, 12])
    df = read_csv(StringIO('a,b\nc\n'), skiprows=0, names=['a'], engine='c')
    tm.assert_frame_equal(df, a)
    df = read_csv(StringIO('1,1,1,1,0\n' * 2 + '\n' * 2), names=list('abcd'), engine='c')
    tm.assert_frame_equal(df, b)
    df = read_csv(StringIO('0,1,2,3,4\n5,6\n7,8,9,10,11\n12,13,14'), names=list('abcd'), engine='c')
    tm.assert_frame_equal(df, c)