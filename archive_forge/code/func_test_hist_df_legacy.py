import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_hist_df_legacy(self, hist_df):
    with tm.assert_produces_warning(UserWarning, check_stacklevel=False):
        _check_plot_works(hist_df.hist)