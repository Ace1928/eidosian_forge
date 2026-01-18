import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_legacy_ax_and_fig(self, ts):
    fig, ax = mpl.pyplot.subplots(1, 1)
    _check_plot_works(ts.hist, ax=ax, figure=fig, default_axes=True)