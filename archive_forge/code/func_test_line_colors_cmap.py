import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('colormap', ['jet', cm.jet])
def test_line_colors_cmap(self, colormap):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot(colormap=colormap)
    rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
    _check_colors(ax.get_lines(), linecolors=rgba_colors)