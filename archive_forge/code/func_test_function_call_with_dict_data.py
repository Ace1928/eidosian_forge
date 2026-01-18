import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('func', all_funcs, ids=all_func_ids)
def test_function_call_with_dict_data(func):
    """Test with dict data -> label comes from the value of 'x' parameter."""
    data = {'a': [1, 2], 'b': [8, 9], 'w': 'NOT'}
    assert func(None, 'a', 'b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func(None, x='a', y='b', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: b'
    assert func(None, 'a', 'b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func(None, 'a', 'b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'
    assert func(None, x='a', y='b', label='', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: '
    assert func(None, x='a', y='b', label='text', data=data) == 'x: [1, 2], y: [8, 9], ls: x, w: xyz, label: text'