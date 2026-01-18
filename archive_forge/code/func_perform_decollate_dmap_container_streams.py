from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def perform_decollate_dmap_container_streams(self, ContainerType):
    xy_stream = XY()
    fn = lambda x, y: Points([x, y])
    data = [(0, DynamicMap(fn, streams=[xy_stream])), (1, DynamicMap(fn, streams=[xy_stream])), (2, DynamicMap(fn, streams=[xy_stream]))]
    container = ContainerType(data, kdims=['c'])
    decollated = container.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(len(decollated.kdims), 0)
    self.assertEqual(len(decollated.streams), 1)
    decollated.streams[0].event(x=2.0, y=3.0)
    xy_stream.event(x=2.0, y=3.0)
    expected_data = [(d[0], d[1][()]) for d in data]
    expected = ContainerType(expected_data, kdims=['c'])
    result = decollated[()]
    self.assertEqual(expected, result)