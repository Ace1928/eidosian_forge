import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_zero_min_count(self):
    tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).opts(min_count=0)
    plot = bokeh_renderer.get_plot(tiles)
    cmapper = plot.handles['color_mapper']
    self.assertEqual(cmapper.low, 0)
    self.assertEqual(plot.state.background_fill_color, cmapper.palette[0])