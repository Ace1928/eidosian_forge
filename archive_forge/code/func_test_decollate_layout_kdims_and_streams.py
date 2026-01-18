from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def test_decollate_layout_kdims_and_streams(self):
    layout = self.dmap_ab + self.dmap_xy
    decollated = layout.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(decollated.kdims, self.dmap_ab.kdims)
    decollated.streams[0].event(x=3.0, y=4.0)
    self.assertEqual(decollated[1.0, 2.0], Points([1.0, 2.0]) + Points([3.0, 4.0]))
    self.assertEqual(decollated.callback.callable(1.0, 2.0, dict(x=3.0, y=4.0)), Points([1.0, 2.0]) + Points([3.0, 4.0]))