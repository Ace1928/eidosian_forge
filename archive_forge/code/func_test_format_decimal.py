import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('formatter', [None, '{:,.4f}'])
@pytest.mark.parametrize('thousands', [None, ',', '*'])
@pytest.mark.parametrize('precision', [None, 4])
@pytest.mark.parametrize('func, col', [('format', 1), ('format_index', 0)])
def test_format_decimal(formatter, thousands, precision, func, col):
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(decimal='_', formatter=formatter, thousands=thousands, precision=precision)._translate(True, True)
    assert '000_123' in result['body'][0][col]['display_value']
    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(decimal='_', formatter=formatter, thousands=thousands, precision=precision)._translate(True, True)
    assert '000_123' in result['body'][0][col]['display_value']