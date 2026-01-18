import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_reindex(self):
    data = [((0, 0.5), 'a'), ((1, 0.5), 'b')]
    ndmap = MultiDimensionalMapping(data, kdims=[self.dim1, self.dim2])
    reduced_dims = ['intdim']
    reduced_ndmap = ndmap.reindex(reduced_dims)
    self.assertEqual([d.name for d in reduced_ndmap.kdims], reduced_dims)