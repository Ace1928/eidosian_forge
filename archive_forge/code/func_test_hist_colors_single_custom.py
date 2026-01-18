import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_hist_colors_single_custom(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    custom_colors = 'rgcby'
    ax = df.plot.hist(color=custom_colors)
    _check_colors(ax.patches[::10], facecolors=custom_colors)