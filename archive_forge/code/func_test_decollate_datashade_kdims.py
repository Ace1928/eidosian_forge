from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
@datashade_skip
def test_decollate_datashade_kdims(self):
    decollated = self.dmap_datashade_kdim_points.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(decollated.kdims, self.dmap_ab.kdims)
    self.assertEqual([PlotSize, RangeXY], [type(s) for s in decollated.streams])
    self.px_stream.event(px=3)
    plot_size, range_xy = self.dmap_datashade_kdim_points.streams
    plot_size.event(width=250, height=300)
    range_xy.event(x_range=(0, 10), y_range=(0, 15))
    expected = self.dmap_datashade_kdim_points[4.0, 5.0]
    result = decollated.callback.callable(4.0, 5.0, {'width': 250, 'height': 300}, {'x_range': (0, 10), 'y_range': (0, 15)})
    self.assertEqual(expected, result)