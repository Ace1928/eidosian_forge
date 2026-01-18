import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
@pytest.mark.slow
@pytest.mark.parametrize('rows, res', [[4, 4], [-1, 3]])
def test_grouped_box_layout_axes_shape_rows(self, hist_df, rows, res):
    df = hist_df
    df.boxplot(column=['height', 'weight', 'category'], by='gender', layout=(rows, 1))
    _check_axes_shape(mpl.pyplot.gcf().axes, axes_num=3, layout=(res, 1))