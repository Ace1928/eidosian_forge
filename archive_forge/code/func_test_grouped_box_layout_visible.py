import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('col, visible', [['height', False], ['weight', True], ['category', True]])
def test_grouped_box_layout_visible(self, hist_df, col, visible):
    df = hist_df
    axes = df.boxplot(column=['height', 'weight', 'category'], by='gender', return_type='axes')
    _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(2, 2))
    ax = axes[col]
    _check_visible(ax.get_xticklabels(), visible=visible)
    _check_visible([ax.xaxis.get_label()], visible=visible)