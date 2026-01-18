import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('level, expected', [(0, ['X', 'X', '_', '_']), ('zero', ['X', 'X', '_', '_']), (1, ['_', '_', 'X', 'X']), ('one', ['_', '_', 'X', 'X']), ([0, 1], ['X', 'X', 'X', 'X']), ([0, 'zero'], ['X', 'X', '_', '_']), ([0, 'one'], ['X', 'X', 'X', 'X']), (['one', 'zero'], ['X', 'X', 'X', 'X'])])
def test_format_index_level(axis, level, expected):
    midx = MultiIndex.from_arrays([['_', '_'], ['_', '_']], names=['zero', 'one'])
    df = DataFrame([[1, 2], [3, 4]])
    if axis == 0:
        df.index = midx
    else:
        df.columns = midx
    styler = df.style.format_index(lambda v: 'X', level=level, axis=axis)
    ctx = styler._translate(True, True)
    if axis == 0:
        result = [ctx['body'][s][0]['display_value'] for s in range(2)]
        result += [ctx['body'][s][1]['display_value'] for s in range(2)]
    else:
        result = [ctx['head'][0][s + 1]['display_value'] for s in range(2)]
        result += [ctx['head'][1][s + 1]['display_value'] for s in range(2)]
    assert expected == result