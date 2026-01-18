from collections import namedtuple
from io import StringIO
import numpy as np
import pytest
from pandas.errors import ParserError
from pandas import (
import pandas._testing as tm
@xfail_pyarrow
@pytest.mark.parametrize('index_col', [None, [0]])
@pytest.mark.parametrize('columns', [None, ['', 'Unnamed'], ['Unnamed', ''], ['Unnamed', 'NotUnnamed']])
def test_multi_index_unnamed(all_parsers, index_col, columns):
    parser = all_parsers
    header = [0, 1]
    if index_col is None:
        data = ','.join(columns or ['', '']) + '\n0,1\n2,3\n4,5\n'
    else:
        data = ','.join([''] + (columns or ['', ''])) + '\n,0,1\n0,2,3\n1,4,5\n'
    result = parser.read_csv(StringIO(data), header=header, index_col=index_col)
    exp_columns = []
    if columns is None:
        columns = ['', '', '']
    for i, col in enumerate(columns):
        if not col:
            col = f'Unnamed: {(i if index_col is None else i + 1)}_level_0'
        exp_columns.append(col)
    columns = MultiIndex.from_tuples(zip(exp_columns, ['0', '1']))
    expected = DataFrame([[2, 3], [4, 5]], columns=columns)
    tm.assert_frame_equal(result, expected)