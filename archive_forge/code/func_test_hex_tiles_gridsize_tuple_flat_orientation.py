import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_gridsize_tuple_flat_orientation(self):
    tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).opts(gridsize=(5, 10), orientation='flat')
    plot = bokeh_renderer.get_plot(tiles)
    glyph = plot.handles['glyph']
    self.assertEqual(glyph.size, 0.13333333333333333)
    self.assertEqual(glyph.aspect_scale, 0.5)