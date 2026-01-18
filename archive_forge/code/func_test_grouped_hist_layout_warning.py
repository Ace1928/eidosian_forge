import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_grouped_hist_layout_warning(self, hist_df):
    df = hist_df
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        axes = _check_plot_works(df.hist, column='height', by=df.gender, layout=(2, 1))
    _check_axes_shape(axes, axes_num=2, layout=(2, 1))