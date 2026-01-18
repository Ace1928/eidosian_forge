import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
@pytest.mark.parametrize('colormap', ['jet', cm.jet])
def test_hist_colors_cmap(self, colormap):
    df = DataFrame(np.random.default_rng(2).standard_normal((5, 5)))
    ax = df.plot.hist(colormap=colormap)
    rgba_colors = [cm.jet(n) for n in np.linspace(0, 1, 5)]
    _check_colors(ax.patches[::10], facecolors=rgba_colors)