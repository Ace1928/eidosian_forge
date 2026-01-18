import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_area_colors(self):
    from matplotlib.collections import PolyCollection
    custom_colors = 'rgcby'
    df = DataFrame(np.random.default_rng(2).random((5, 5)))
    ax = df.plot.area(color=custom_colors)
    _check_colors(ax.get_lines(), linecolors=custom_colors)
    poly = [o for o in ax.get_children() if isinstance(o, PolyCollection)]
    _check_colors(poly, facecolors=custom_colors)
    handles, _ = ax.get_legend_handles_labels()
    _check_colors(handles, facecolors=custom_colors)
    for h in handles:
        assert h.get_alpha() is None