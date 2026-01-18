import numpy as np
from bokeh.layouts import Column
from bokeh.models import Div, Toolbar
from holoviews.core import (
from holoviews.element import Curve, Image, Points
from holoviews.operation import gridmatrix
from holoviews.streams import Stream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_gridspace_sparse(self):
    grid = GridSpace({(i, j): Curve(range(i + j)) for i in range(1, 3) for j in range(2, 4) if not (i == 1 and j == 2)})
    plot = bokeh_renderer.get_plot(grid)
    size = bokeh_renderer.get_size(plot.state)
    self.assertEqual(size, (320, 311))