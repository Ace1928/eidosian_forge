import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('layout_test', ({'layout': None, 'expected_size': (2, 2)}, {'layout': (2, 2), 'expected_size': (2, 2)}, {'layout': (4, 1), 'expected_size': (4, 1)}, {'layout': (1, 4), 'expected_size': (1, 4)}, {'layout': (3, 3), 'expected_size': (3, 3)}, {'layout': (-1, 4), 'expected_size': (1, 4)}, {'layout': (4, -1), 'expected_size': (4, 1)}, {'layout': (-1, 2), 'expected_size': (2, 2)}, {'layout': (2, -1), 'expected_size': (2, 2)}))
def test_hist_layout(self, layout_test):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)))
    df[2] = to_datetime(np.random.default_rng(2).integers(812419200000000000, 819331200000000000, size=10, dtype=np.int64))
    axes = df.hist(layout=layout_test['layout'])
    expected = layout_test['expected_size']
    _check_axes_shape(axes, axes_num=3, layout=expected)