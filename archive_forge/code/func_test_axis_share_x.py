import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_axis_share_x(self, hist_df):
    df = hist_df
    ax1, ax2 = df.hist(column='height', by=df.gender, sharex=True)
    assert get_x_axis(ax1).joined(ax1, ax2)
    assert get_x_axis(ax2).joined(ax1, ax2)
    assert not get_y_axis(ax1).joined(ax1, ax2)
    assert not get_y_axis(ax2).joined(ax1, ax2)