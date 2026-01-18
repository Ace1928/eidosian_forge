import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_legacy_fig(self, ts):
    fig, _ = mpl.pyplot.subplots(1, 1)
    _check_plot_works(ts.hist, figure=fig, default_axes=True)