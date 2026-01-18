import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('layout, check_layout, figsize', [[(4, 1), (4, 1), None], [(-1, 1), (4, 1), None], [(4, 2), (4, 2), (12, 8)]])
def test_grouped_hist_layout_figsize(self, hist_df, layout, check_layout, figsize):
    df = hist_df
    axes = df.hist(column='height', by=df.category, layout=layout, figsize=figsize)
    _check_axes_shape(axes, axes_num=4, layout=check_layout, figsize=figsize)