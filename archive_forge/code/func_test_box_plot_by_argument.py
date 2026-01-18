import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('by, column, titles, xticklabels', [('C', 'A', ['A'], [['a', 'b', 'c']]), (['C', 'D'], 'A', ['A'], [['(a, a)', '(b, b)', '(c, c)']]), ('C', ['A', 'B'], ['A', 'B'], [['a', 'b', 'c']] * 2), (['C', 'D'], ['A', 'B'], ['A', 'B'], [['(a, a)', '(b, b)', '(c, c)']] * 2), (['C'], None, ['A', 'B'], [['a', 'b', 'c']] * 2)])
def test_box_plot_by_argument(self, by, column, titles, xticklabels, hist_df):
    axes = _check_plot_works(hist_df.plot.box, default_axes=True, column=column, by=by)
    result_titles = [ax.get_title() for ax in axes]
    result_xticklabels = [[label.get_text() for label in ax.get_xticklabels()] for ax in axes]
    assert result_xticklabels == xticklabels
    assert result_titles == titles