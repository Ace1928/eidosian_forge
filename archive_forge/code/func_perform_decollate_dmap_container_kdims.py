from unittest import skipIf
import param
from holoviews.core import DynamicMap, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import PlotSize, RangeXY, Stream
def perform_decollate_dmap_container_kdims(self, ContainerType):
    data = [(0, self.dmap_ab.clone()), (1, self.dmap_ab.clone()), (2, self.dmap_ab.clone())]
    container = ContainerType(data, kdims=['c'])
    decollated = container.decollate()
    self.assertIsInstance(decollated, DynamicMap)
    self.assertEqual(decollated.kdims, self.dmap_ab.kdims)
    a, b = (2.0, 3.0)
    expected_data = [(d[0], d[1][a, b]) for d in data]
    expected = ContainerType(expected_data, kdims=['c'])
    result = decollated[a, b]
    self.assertEqual(expected, result)