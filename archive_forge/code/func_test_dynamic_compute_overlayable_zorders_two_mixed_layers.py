import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dynamic_compute_overlayable_zorders_two_mixed_layers(self):
    area = Area(range(10))
    dmap = DynamicMap(lambda: Curve(range(10)), kdims=[])
    combined = area * dmap
    combined[()]
    sources = compute_overlayable_zorders(combined)
    self.assertEqual(sources[0], [area])
    self.assertEqual(sources[1], [dmap])