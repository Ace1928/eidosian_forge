import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_empty(self):
    tiles = HexTiles([])
    plot = bokeh_renderer.get_plot(tiles)
    self.assertEqual(plot.handles['source'].data, {'q': [], 'r': []})