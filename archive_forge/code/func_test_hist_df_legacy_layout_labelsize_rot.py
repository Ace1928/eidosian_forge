import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_hist_df_legacy_layout_labelsize_rot(self, frame_or_series):
    obj = frame_or_series(range(10))
    xf, yf = (20, 18)
    xrot, yrot = (30, 40)
    axes = obj.hist(xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)
    _check_ticks_props(axes, xlabelsize=xf, xrot=xrot, ylabelsize=yf, yrot=yrot)