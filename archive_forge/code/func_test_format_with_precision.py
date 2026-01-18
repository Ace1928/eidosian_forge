import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('precision, expected', [(1, ['1.0', '2.0', '3.2', '4.6']), (2, ['1.00', '2.01', '3.21', '4.57']), (3, ['1.000', '2.009', '3.212', '4.566'])])
def test_format_with_precision(precision, expected):
    df = DataFrame([[1.0, 2.009, 3.2121, 4.566]], columns=[1.0, 2.009, 3.2121, 4.566])
    styler = Styler(df)
    styler.format(precision=precision)
    styler.format_index(precision=precision, axis=1)
    ctx = styler._translate(True, True)
    for col, exp in enumerate(expected):
        assert ctx['body'][0][col + 1]['display_value'] == exp
        assert ctx['head'][0][col + 1]['display_value'] == exp