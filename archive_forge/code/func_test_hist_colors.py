import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_hist_colors(self):
    default_colors = _unpack_cycler(mpl.pyplot.rcParams)
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot.hist()
    _check_colors(ax.patches[::10], facecolors=default_colors[:5])