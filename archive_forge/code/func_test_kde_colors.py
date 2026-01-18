import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_kde_colors(self):
    pytest.importorskip('scipy')
    custom_colors = 'rgcby'
    df = DataFrame(np.random.default_rng(2).random((5, 5)))
    ax = df.plot.kde(color=custom_colors)
    _check_colors(ax.get_lines(), linecolors=custom_colors)