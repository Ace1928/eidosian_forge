import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_apply_key_type(self):
    data = dict([(0.5, 'a'), (1.5, 'b')])
    ndmap = MultiDimensionalMapping(data, kdims=[self.dim1])
    self.assertEqual(list(ndmap.keys()), [0, 1])