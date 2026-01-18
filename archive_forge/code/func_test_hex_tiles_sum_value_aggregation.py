import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_sum_value_aggregation(self):
    tiles = HexTiles([(0, 0, 1), (0.5, 0.5, 2), (-0.5, -0.5, 3), (-0.4, -0.4, 4)], vdims='z')
    binned = hex_binning(tiles, gridsize=3, aggregator=np.sum)
    expected = HexTiles([(0, 0, 1), (2, -1, 2), (-2, 1, 7)], kdims=[Dimension('x', range=(-0.5, 0.5)), Dimension('y', range=(-0.5, 0.5))], vdims='z')
    self.assertEqual(binned, expected)