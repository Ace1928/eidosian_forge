import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_hist_subplot_xrot(self):
    df = DataFrame({'length': [1.5, 0.5, 1.2, 0.9, 3], 'animal': ['pig', 'rabbit', 'pig', 'pig', 'rabbit']})
    axes = _check_plot_works(df.hist, default_axes=True, column='length', by='animal', bins=5, xrot=0)
    _check_ticks_props(axes, xrot=0)