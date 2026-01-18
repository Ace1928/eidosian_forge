import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_get_range_from_element(self):
    dim = Dimension('y', soft_range=(0, 3), range=(0, 2))
    element = Scatter([1, 2, 3], vdims=dim)
    drange, srange, hrange = get_range(element, {}, dim)
    self.assertEqual(drange, (1, 3))
    self.assertEqual(srange, (0, 3))
    self.assertEqual(hrange, (0, 2))