import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('gb_key, axes_num, rows', [['gender', 2, 1], ['category', 4, 2], ['classroom', 3, 2]])
def test_grouped_box_layout_positive_layout_axes(self, hist_df, gb_key, axes_num, rows):
    df = hist_df
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        _check_plot_works(df.groupby(gb_key).boxplot, column='height', return_type='dict')
    _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=axes_num, layout=(rows, 2))