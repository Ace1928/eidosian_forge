import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
@pytest.mark.parametrize('styles,expected', [([('color', 'green'), ('color', 'red')], 'color: red;'), ([('font-weight', 'bold'), ('font-weight', 'normal')], 'font-weight: normal;'), ([('text-align', 'center'), ('TEXT-ALIGN', 'right')], 'text-align: right;')])
def test_css_excel_cell_precedence(styles, expected):
    """It applies favors latter declarations over former declarations"""
    converter = CSSToExcelConverter()
    converter._call_cached.cache_clear()
    css_styles = {(0, 0): styles}
    cell = CssExcelCell(row=0, col=0, val='', style=None, css_styles=css_styles, css_row=0, css_col=0, css_converter=converter)
    converter._call_cached.cache_clear()
    assert cell.style == converter(expected)