import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('msg, by, layout', [('larger than required size', ['C', 'D'], (1, 1)), (re.escape('Layout must be a tuple of (rows, columns)'), 'C', (1,)), ('At least one dimension of layout must be positive', 'C', (-1, -1))])
def test_box_plot_invalid_layout_with_by_raises(self, msg, by, layout, hist_df):
    with pytest.raises(ValueError, match=msg):
        hist_df.plot.box(column=['A', 'B'], by=by, layout=layout)