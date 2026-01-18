import numpy as np
from bokeh.layouts import Column
from bokeh.models import Div, Toolbar
from holoviews.core import (
from holoviews.element import Curve, Image, Points
from holoviews.operation import gridmatrix
from holoviews.streams import Stream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_grid_title_update(self):
    grid = GridSpace({(i, j): HoloMap({a: Image(np.random.rand(10, 10)) for a in range(3)}, kdims=['X']) for i in range(2) for j in range(3)})
    plot = bokeh_renderer.get_plot(grid)
    plot.update(1)
    title = plot.handles['title']
    self.assertIsInstance(title, Div)
    text = '<span style="color:black;font-family:Arial;font-style:bold;font-weight:bold;font-size:16pt">X: 1</span>'
    self.assertEqual(title.text, text)