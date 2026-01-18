import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
def test_function_call_replace_all():
    """Test without a "replace_names" argument, all vars should be replaced."""
    data = {'a': [1, 2], 'b': [8, 9], 'x': 'xyz'}

    @_preprocess_data(label_namer='y')
    def func_replace_all(ax, x, y, ls='x', label=None, w='NOT'):
        return f'x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}'
    assert func_replace_all(None, 'a', 'b', w='x', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func_replace_all(None, x='a', y='b', w='x', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func_replace_all(None, 'a', 'b', w='x', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func_replace_all(None, 'a', 'b', w='x', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'
    assert func_replace_all(None, x='a', y='b', w='x', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func_replace_all(None, x='a', y='b', w='x', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'