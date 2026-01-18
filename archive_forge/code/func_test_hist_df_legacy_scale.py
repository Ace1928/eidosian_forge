import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
@pytest.mark.slow
def test_hist_df_legacy_scale(self):
    ser = Series(range(10))
    ax = ser.hist(log=True)
    _check_ax_scales(ax, yaxis='log')