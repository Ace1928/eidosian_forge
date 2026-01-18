import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_axis_share_y_with_by(self, hist_df):
    ax1, ax2, ax3 = hist_df.plot.hist(column='A', by='C', sharey=True)
    assert get_y_axis(ax1).joined(ax1, ax2)
    assert get_y_axis(ax2).joined(ax1, ax2)
    assert get_y_axis(ax3).joined(ax1, ax3)
    assert get_y_axis(ax3).joined(ax2, ax3)
    assert not get_x_axis(ax1).joined(ax1, ax2)
    assert not get_x_axis(ax2).joined(ax1, ax2)
    assert not get_x_axis(ax3).joined(ax1, ax3)
    assert not get_x_axis(ax3).joined(ax2, ax3)