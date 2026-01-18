import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_holomap_map_with_none(self):
    hmap = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z'])
    mapped = hmap.map(lambda x: x if x.range(1)[1] > 0 else None, Dataset)
    self.assertEqual(hmap[1:10], mapped)