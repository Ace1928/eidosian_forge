import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('color', ['rgcby', list('rgcby')])
def test_line_colors_and_styles_subplots_custom_colors(self, color):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    axes = df.plot(color=color, subplots=True)
    for ax, c in zip(axes, list(color)):
        _check_colors(ax.get_lines(), linecolors=[c])