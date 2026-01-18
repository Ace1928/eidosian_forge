import os
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_bar_plt_xaxis_intervalrange(self):
    from matplotlib.text import Text
    expected = [Text(0, 0, '([0, 1],)'), Text(1, 0, '([1, 2],)')]
    s = Series([1, 2], index=[interval_range(0, 2, closed='both')])
    _check_plot_works(s.plot.bar)
    assert all((a.get_text() == b.get_text() for a, b in zip(s.plot.bar().get_xticklabels(), expected)))