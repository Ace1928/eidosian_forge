import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('figsize', [(12, 8), (20, 10)])
def test_figure_shape_hist_with_by(self, figsize, hist_df):
    axes = hist_df.plot.box(column='A', by='C', figsize=figsize)
    _check_axes_shape(axes, axes_num=1, figsize=figsize)