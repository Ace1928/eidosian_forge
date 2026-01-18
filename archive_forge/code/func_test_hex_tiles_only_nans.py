import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_only_nans(self):
    tiles = HexTiles([(np.nan, 0), (1, np.nan)])
    plot = bokeh_renderer.get_plot(tiles)
    self.assertEqual(plot.handles['source'].data, {'q': [], 'r': []})