import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_no_overlap(self):
    from matplotlib.pyplot import gcf, subplot
    x = Series(np.random.default_rng(2).standard_normal(2))
    y = Series(np.random.default_rng(2).standard_normal(2))
    subplot(121)
    x.hist()
    subplot(122)
    y.hist()
    fig = gcf()
    axes = fig.axes
    assert len(axes) == 2