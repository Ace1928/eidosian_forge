import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
def test_empty_group(self):
    """Test to prevent regression of issue fixed in #5131"""
    ls = np.linspace(0, 10, 200)
    xx, yy = np.meshgrid(ls, ls)
    util.render(Image(np.sin(xx) * np.cos(yy), group='').opts(cmap='greys'))