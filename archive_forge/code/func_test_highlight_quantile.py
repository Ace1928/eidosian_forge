import numpy as np
import pytest
from pandas import (
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('kwargs', [{'q_left': 0.5, 'q_right': 1, 'axis': 0}, {'q_left': 0.5, 'q_right': 1, 'axis': None}, {'q_left': 0, 'q_right': 1, 'subset': IndexSlice[2, :]}, {'q_left': 0.5, 'axis': 0}, {'q_right': 1, 'subset': IndexSlice[2, :], 'axis': 1}, {'q_left': 0.5, 'axis': 0, 'props': 'background-color: yellow'}])
def test_highlight_quantile(styler, kwargs):
    expected = {(2, 0): [('background-color', 'yellow')], (2, 1): [('background-color', 'yellow')]}
    result = styler.highlight_quantile(**kwargs)._compute().ctx
    assert result == expected