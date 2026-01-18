import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import _str_escape
@pytest.mark.parametrize('formatter', [None, '{:,.1f}'])
@pytest.mark.parametrize('decimal', ['.', '*'])
@pytest.mark.parametrize('precision', [None, 2])
@pytest.mark.parametrize('func, col', [('format', 1), ('format_index', 0)])
def test_format_thousands(formatter, decimal, precision, func, col):
    styler = DataFrame([[1000000.123456789]], index=[1000000.123456789]).style
    result = getattr(styler, func)(thousands='_', formatter=formatter, decimal=decimal, precision=precision)._translate(True, True)
    assert '1_000_000' in result['body'][0][col]['display_value']
    styler = DataFrame([[1000000]], index=[1000000]).style
    result = getattr(styler, func)(thousands='_', formatter=formatter, decimal=decimal, precision=precision)._translate(True, True)
    assert '1_000_000' in result['body'][0][col]['display_value']
    styler = DataFrame([[1 + 1000000.123456789j]], index=[1 + 1000000.123456789j]).style
    result = getattr(styler, func)(thousands='_', formatter=formatter, decimal=decimal, precision=precision)._translate(True, True)
    assert '1_000_000' in result['body'][0][col]['display_value']