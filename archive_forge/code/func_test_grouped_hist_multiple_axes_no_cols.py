import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_grouped_hist_multiple_axes_no_cols(self, hist_df):
    df = hist_df
    fig, axes = mpl.pyplot.subplots(2, 3)
    returned = df.hist(by='classroom', ax=axes[1])
    _check_axes_shape(returned, axes_num=3, layout=(1, 3))
    tm.assert_numpy_array_equal(returned, axes[1])
    assert returned[0].figure is fig