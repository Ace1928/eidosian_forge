import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('by, column, titles, xticklabels', [(0, 'A', ['A'], [['a', 'b', 'c']]), ([0, 'D'], 'A', ['A'], [['(a, a)', '(b, b)', '(c, c)']]), (0, None, ['A', 'B'], [['a', 'b', 'c']] * 2)])
def test_box_plot_by_0(self, by, column, titles, xticklabels, hist_df):
    df = hist_df.copy()
    df = df.rename(columns={'C': 0})
    axes = _check_plot_works(df.plot.box, default_axes=True, column=column, by=by)
    result_titles = [ax.get_title() for ax in axes]
    result_xticklabels = [[label.get_text() for label in ax.get_xticklabels()] for ax in axes]
    assert result_xticklabels == xticklabels
    assert result_titles == titles