import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_holomap_redim_nested(self):
    hmap = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z'])
    redimmed = hmap.redim(x='Time', z='Magnitude')
    self.assertEqual(redimmed.dimensions('all', True), ['Magnitude', 'Time', 'y'])