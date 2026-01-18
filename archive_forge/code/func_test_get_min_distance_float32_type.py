import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_get_min_distance_float32_type(self):
    xs, ys = (np.arange(0, 2.0, 0.2, dtype='float32'), np.arange(0, 2.0, 0.2, dtype='float32'))
    X, Y = np.meshgrid(xs, ys)
    dist = get_min_distance(Points((X.flatten(), Y.flatten())))
    self.assertEqual(float(round(dist, 5)), 0.2)