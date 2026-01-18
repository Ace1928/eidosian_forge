import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_dataframe_none_color(self):
    df = DataFrame([[1, 2, 3]])
    ax = df.plot(color=None)
    expected = _unpack_cycler(mpl.pyplot.rcParams)[:3]
    _check_colors(ax.get_lines(), linecolors=expected)