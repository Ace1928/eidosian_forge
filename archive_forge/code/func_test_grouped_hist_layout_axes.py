import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
@pytest.mark.parametrize('kwargs, axes_num, layout', [[{'by': 'gender', 'layout': (3, 5)}, 2, (3, 5)], [{'column': ['height', 'weight', 'category']}, 3, (2, 2)]])
def test_grouped_hist_layout_axes(self, hist_df, kwargs, axes_num, layout):
    df = hist_df
    axes = df.hist(**kwargs)
    _check_axes_shape(axes, axes_num=axes_num, layout=layout)