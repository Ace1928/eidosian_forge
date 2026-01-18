import gc
import numpy as np
import pytest
from pandas import (
import matplotlib as mpl
from pandas.io.formats.style import Styler
@pytest.mark.parametrize('axis, expected', [(0, ['low', 'low', 'high', 'high']), (1, ['low', 'high', 'low', 'high']), (None, ['low', 'mid', 'mid', 'high'])])
@pytest.mark.parametrize('f', ['background_gradient', 'text_gradient'])
def test_background_gradient_axis(styler, axis, expected, f):
    if f == 'background_gradient':
        colors = {'low': [('background-color', '#f7fbff'), ('color', '#000000')], 'mid': [('background-color', '#abd0e6'), ('color', '#000000')], 'high': [('background-color', '#08306b'), ('color', '#f1f1f1')]}
    elif f == 'text_gradient':
        colors = {'low': [('color', '#f7fbff')], 'mid': [('color', '#abd0e6')], 'high': [('color', '#08306b')]}
    result = getattr(styler, f)(cmap='Blues', axis=axis)._compute().ctx
    for i, cell in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        assert result[cell] == colors[expected[i]]