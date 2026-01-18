import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_bar_colors(self):
    default_colors = _unpack_cycler(plt.rcParams)
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot.bar()
    _check_colors(ax.patches[::5], facecolors=default_colors[:5])