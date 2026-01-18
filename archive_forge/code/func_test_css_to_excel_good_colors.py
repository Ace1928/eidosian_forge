import string
import pytest
from pandas.errors import CSSWarning
import pandas._testing as tm
from pandas.io.formats.excel import (
@pytest.mark.parametrize('input_color,output_color', list(CSSToExcelConverter.NAMED_COLORS.items()) + [('#' + rgb, rgb) for rgb in CSSToExcelConverter.NAMED_COLORS.values()] + [('#F0F', 'FF00FF'), ('#ABC', 'AABBCC')])
def test_css_to_excel_good_colors(input_color, output_color):
    css = f'border-top-color: {input_color}; border-right-color: {input_color}; border-bottom-color: {input_color}; border-left-color: {input_color}; background-color: {input_color}; color: {input_color}'
    expected = {}
    expected['fill'] = {'patternType': 'solid', 'fgColor': output_color}
    expected['font'] = {'color': output_color}
    expected['border'] = {k: {'color': output_color, 'style': 'none'} for k in ('top', 'right', 'bottom', 'left')}
    with tm.assert_produces_warning(None):
        convert = CSSToExcelConverter()
        assert expected == convert(css)