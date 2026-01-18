import numpy as np
from bokeh.layouts import Column
from bokeh.models import Div, Toolbar
from holoviews.core import (
from holoviews.element import Curve, Image, Points
from holoviews.operation import gridmatrix
from holoviews.streams import Stream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_grid_dimensioned_stream_title_update(self):
    stream = Stream.define('Test', test=0)()
    dmap = DynamicMap(lambda test: Curve([]), kdims=['test'], streams=[stream])
    grid = GridMatrix({0: dmap, 1: Curve([])}, 'X')
    plot = bokeh_renderer.get_plot(grid)
    self.assertIn('test: 0', plot.handles['title'].text)
    stream.event(test=1)
    self.assertIn('test: 1', plot.handles['title'].text)
    plot.cleanup()
    self.assertEqual(stream._subscribers, [])