import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('by, column, layout, axes_num', [(['C'], 'A', (1, 1), 1), ('C', 'A', (1, 1), 1), ('C', None, (2, 1), 2), ('C', ['A', 'B'], (1, 2), 2), (['C', 'D'], 'A', (1, 1), 1), (['C', 'D'], None, (1, 2), 2)])
def test_box_plot_layout_with_by(self, by, column, layout, axes_num, hist_df):
    axes = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by, layout=layout)
    _check_axes_shape(axes, axes_num=axes_num, layout=layout)