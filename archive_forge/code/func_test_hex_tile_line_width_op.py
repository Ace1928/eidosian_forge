import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tile_line_width_op(self):
    hextiles = HexTiles(np.random.randn(1000, 2)).opts(line_width='Count')
    plot = bokeh_renderer.get_plot(hextiles)
    glyph = plot.handles['glyph']
    self.assertEqual(property_to_dict(glyph.line_width), {'field': 'line_width'})