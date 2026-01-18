import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('f', ['highlight_min', 'highlight_max'])
@pytest.mark.parametrize('kwargs', [{'axis': None, 'color': 'red'}, {'axis': 0, 'subset': ['A'], 'color': 'red'}, {'axis': None, 'props': 'background-color: red'}])
def test_highlight_minmax_ext(df, f, kwargs):
    expected = {(2, 0): [('background-color', 'red')]}
    if f == 'highlight_min':
        df = -df
    result = getattr(df.style, f)(**kwargs)._compute().ctx
    assert result == expected