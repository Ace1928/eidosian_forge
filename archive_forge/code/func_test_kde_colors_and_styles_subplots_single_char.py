import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_kde_colors_and_styles_subplots_single_char(self):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    axes = df.plot(kind='kde', style='r', subplots=True)
    for ax in axes:
        _check_colors(ax.get_lines(), linecolors=['r'])