import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_count_aggregation(self):
    tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)])
    binned = hex_binning(tiles, gridsize=3)
    expected = HexTiles([(0, 0, 1), (2, -1, 1), (-2, 1, 2)], kdims=[Dimension('x', range=(-0.5, 0.5)), Dimension('y', range=(-0.5, 0.5))], vdims='Count')
    self.assertEqual(binned, expected)