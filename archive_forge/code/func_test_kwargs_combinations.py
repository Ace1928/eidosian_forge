from textwrap import dedent
import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
from pandas.io.formats.style_render import (
@pytest.mark.parametrize('label', [(None, ''), ('text', '\\label{text}')])
@pytest.mark.parametrize('position', [(None, ''), ('h!', '{table}[h!]')])
@pytest.mark.parametrize('caption', [(None, ''), ('text', '\\caption{text}')])
@pytest.mark.parametrize('column_format', [(None, ''), ('rcrl', '{tabular}{rcrl}')])
@pytest.mark.parametrize('position_float', [(None, ''), ('centering', '\\centering')])
def test_kwargs_combinations(styler, label, position, caption, column_format, position_float):
    result = styler.to_latex(label=label[0], position=position[0], caption=caption[0], column_format=column_format[0], position_float=position_float[0])
    assert label[1] in result
    assert position[1] in result
    assert caption[1] in result
    assert column_format[1] in result
    assert position_float[1] in result