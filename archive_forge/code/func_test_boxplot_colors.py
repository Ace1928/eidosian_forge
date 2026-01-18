import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_boxplot_colors(self):
    default_colors = _unpack_cycler(mpl.pyplot.rcParams)
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    bp = df.plot.box(return_type='dict')
    _check_colors_box(bp, default_colors[0], default_colors[0], default_colors[2], default_colors[0])