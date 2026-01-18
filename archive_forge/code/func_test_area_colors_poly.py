import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_area_colors_poly(self):
    from matplotlib import cm
    from matplotlib.collections import PolyCollection
    df = DataFrame(np.random.default_rng(2).random((5, 5)))
    ax = df.plot.area(colormap='jet')
    jet_colors = [cm.jet(n) for n in np.linspace(0, 1, len(df))]
    _check_colors(ax.get_lines(), linecolors=jet_colors)
    poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
    _check_colors(poly, facecolors=jet_colors)
    handles, _ = ax.get_legend_handles_labels()
    _check_colors(handles, facecolors=jet_colors)
    for h in handles:
        assert h.get_alpha() is None