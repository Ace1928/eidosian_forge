import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('by, column, layout, axes_num', [(['C'], 'A', (2, 2), 3), ('C', 'A', (2, 2), 3), (['C'], ['A'], (1, 3), 3), ('C', None, (3, 1), 3), ('C', ['A', 'B'], (3, 1), 3), (['C', 'D'], 'A', (9, 1), 3), (['C', 'D'], 'A', (3, 3), 3), (['C', 'D'], ['A'], (5, 2), 3), (['C', 'D'], ['A', 'B'], (9, 1), 3), (['C', 'D'], None, (9, 1), 3), (['C', 'D'], ['A', 'B'], (5, 2), 3)])
def test_hist_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        axes = _check_plot_works(hist_df.plot.hist, column=column, by=by, layout=layout)
    _check_axes_shape(axes, axes_num=axes_num, layout=layout)