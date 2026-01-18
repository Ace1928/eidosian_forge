import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('kwargs', [{'left': 0, 'right': 1}, {'left': 0, 'right': 1, 'props': 'background-color: yellow'}, {'left': -100, 'right': 100, 'subset': IndexSlice[[0, 1], :]}, {'left': 0, 'subset': IndexSlice[[0, 1], :]}, {'right': 1}, {'left': [0, 0, 11], 'axis': 0}, {'left': DataFrame({'A': [0, 0, 11], 'B': [1, 1, 11]}), 'axis': None}, {'left': 0, 'right': [0, 1], 'axis': 1}])
def test_highlight_between(styler, kwargs):
    expected = {(0, 0): [('background-color', 'yellow')], (0, 1): [('background-color', 'yellow')]}
    result = styler.highlight_between(**kwargs)._compute().ctx
    assert result == expected