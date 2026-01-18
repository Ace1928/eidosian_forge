import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_layout_with_by_shape(self, hist_df):
    df = hist_df
    axes = df.height.hist(by=df.category, layout=(4, 2), figsize=(12, 7))
    _check_axes_shape(axes, axes_num=4, layout=(4, 2), figsize=(12, 7))