import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
@pytest.mark.parametrize('styles,cache_hits,cache_misses', [([[('color', 'green'), ('color', 'red'), ('color', 'green')]], 0, 1), ([[('font-weight', 'bold')], [('font-weight', 'normal'), ('font-weight', 'bold')]], 1, 1), ([[('text-align', 'center')], [('TEXT-ALIGN', 'center')]], 1, 1), ([[('font-weight', 'bold'), ('text-align', 'center')], [('font-weight', 'bold'), ('text-align', 'left')]], 0, 2), ([[('font-weight', 'bold'), ('text-align', 'center')], [('font-weight', 'bold'), ('text-align', 'left')], [('font-weight', 'bold'), ('text-align', 'center')]], 1, 2)])
def test_css_excel_cell_cache(styles, cache_hits, cache_misses):
    """It caches unique cell styles"""
    converter = CSSToExcelConverter()
    converter._call_cached.cache_clear()
    css_styles = {(0, i): _style for i, _style in enumerate(styles)}
    for css_row, css_col in css_styles:
        CssExcelCell(row=0, col=0, val='', style=None, css_styles=css_styles, css_row=css_row, css_col=css_col, css_converter=converter)
    cache_info = converter._call_cached.cache_info()
    converter._call_cached.cache_clear()
    assert cache_info.hits == cache_hits
    assert cache_info.misses == cache_misses