import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_get_range_from_ranges(self):
    dim = Dimension('y', soft_range=(0, 3), range=(0, 2))
    element = Scatter([1, 2, 3], vdims=dim)
    ranges = {'y': {'soft': (-1, 4), 'hard': (-1, 3), 'data': (-0.5, 2.5)}}
    drange, srange, hrange = get_range(element, ranges, dim)
    self.assertEqual(drange, (-0.5, 2.5))
    self.assertEqual(srange, (-1, 4))
    self.assertEqual(hrange, (-1, 3))