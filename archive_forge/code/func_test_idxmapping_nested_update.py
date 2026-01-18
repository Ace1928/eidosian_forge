import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_nested_update(self):
    data1 = [(0, 'a'), (1, 'b')]
    data2 = [(2, 'c'), (3, 'd')]
    data3 = [(2, 'e'), (3, 'f')]
    ndmap1 = MultiDimensionalMapping(data1, kdims=[self.dim1])
    ndmap2 = MultiDimensionalMapping(data2, kdims=[self.dim1])
    ndmap3 = MultiDimensionalMapping(data3, kdims=[self.dim1])
    ndmap_list = [(0.5, ndmap1), (1.5, ndmap2)]
    nested_ndmap = MultiDimensionalMapping(ndmap_list, kdims=[self.dim2])
    nested_ndmap[0.5,].update(dict([(0, 'c'), (1, 'd')]))
    self.assertEqual(list(nested_ndmap[0.5].values()), ['c', 'd'])
    nested_ndmap[1.5] = ndmap3
    self.assertEqual(list(nested_ndmap[1.5].values()), ['e', 'f'])