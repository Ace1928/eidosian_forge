import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('by, column, titles, legends', [(0, 'A', ['a', 'b', 'c'], [['A']] * 3), (0, None, ['a', 'b', 'c'], [['A', 'B']] * 3), ([0, 'D'], 'A', ['(a, a)', '(b, b)', '(c, c)'], [['A']] * 3)])
def test_hist_plot_by_0(self, by, column, titles, legends, hist_df):
    df = hist_df.copy()
    df = df.rename(columns={'C': 0})
    axes = _check_plot_works(df.plot.hist, default_axes=True, column=column, by=by)
    result_titles = [ax.get_title() for ax in axes]
    result_legends = [[legend.get_text() for legend in ax.get_legend().texts] for ax in axes]
    assert result_legends == legends
    assert result_titles == titles