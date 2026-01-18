import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_by_no_extra_plots(self, hist_df):
    df = hist_df
    df.height.hist(by=df.gender)
    assert len(mpl.pyplot.get_fignums()) == 1