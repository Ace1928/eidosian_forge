import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_hist_df_legacy_rectangles(self):
    from matplotlib.patches import Rectangle
    ser = Series(range(10))
    ax = ser.hist(cumulative=True, bins=4, density=True)
    rects = [x for x in ax.get_children() if isinstance(x, Rectangle)]
    tm.assert_almost_equal(rects[-1].get_height(), 1.0)