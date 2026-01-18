import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_kwargs(self, ts):
    _, ax = mpl.pyplot.subplots()
    ax = ts.plot.hist(bins=5, ax=ax)
    assert len(ax.patches) == 5
    _check_text_labels(ax.yaxis.get_label(), 'Frequency')