import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_dynamic_compute_overlayable_zorders_two_deep_dynamic_layers(self):
    area = DynamicMap(lambda: Area(range(10)), kdims=[])
    curve = DynamicMap(lambda: Curve(range(10)), kdims=[])
    area_redim = area.redim(x='x2')
    curve_redim = curve.redim(x='x2')
    combined = area_redim * curve_redim
    combined[()]
    sources = compute_overlayable_zorders(combined)
    self.assertIn(area_redim, sources[0])
    self.assertIn(area, sources[0])
    self.assertNotIn(curve_redim, sources[0])
    self.assertNotIn(curve, sources[0])
    self.assertIn(curve_redim, sources[1])
    self.assertIn(curve, sources[1])
    self.assertNotIn(area_redim, sources[1])
    self.assertNotIn(area, sources[1])