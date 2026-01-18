import re
import numpy as np
import pytest
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.parametrize('by, column', [([], ['A']), ([], ['A', 'B']), ((), None), ((), ['A', 'B'])])
def test_hist_plot_empty_list_string_tuple_by(self, by, column, hist_df):
    msg = 'No group keys passed'
    with pytest.raises(ValueError, match=msg):
        _check_plot_works(hist_df.plot.hist, default_axes=True, column=column, by=by)