from io import StringIO
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@skip_pyarrow
@pytest.mark.parametrize('index_col,kwargs', [(None, {'columns': ['x', 'y', 'z']}), (False, {'columns': ['x', 'y', 'z']}), (0, {'columns': ['y', 'z'], 'index': Index([], name='x')}), (1, {'columns': ['x', 'z'], 'index': Index([], name='y')}), ('x', {'columns': ['y', 'z'], 'index': Index([], name='x')}), ('y', {'columns': ['x', 'z'], 'index': Index([], name='y')}), ([0, 1], {'columns': ['z'], 'index': MultiIndex.from_arrays([[]] * 2, names=['x', 'y'])}), (['x', 'y'], {'columns': ['z'], 'index': MultiIndex.from_arrays([[]] * 2, names=['x', 'y'])}), ([1, 0], {'columns': ['z'], 'index': MultiIndex.from_arrays([[]] * 2, names=['y', 'x'])}), (['y', 'x'], {'columns': ['z'], 'index': MultiIndex.from_arrays([[]] * 2, names=['y', 'x'])})])
def test_index_col_empty_data(all_parsers, index_col, kwargs):
    data = 'x,y,z'
    parser = all_parsers
    result = parser.read_csv(StringIO(data), index_col=index_col)
    expected = DataFrame(**kwargs)
    tm.assert_frame_equal(result, expected)