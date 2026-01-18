import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_kde_colors_and_styles_subplots_custom_color(self):
    pytest.importorskip('scipy')
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    custom_colors = 'rgcby'
    axes = df.plot(kind='kde', color=custom_colors, subplots=True)
    for ax, c in zip(axes, list(custom_colors)):
        _check_colors(ax.get_lines(), linecolors=[c])