import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_ndims(self):
    dims = [self.dim1, self.dim2, 'strdim']
    idxmap = MultiDimensionalMapping(kdims=dims)
    self.assertEqual(idxmap.ndims, len(dims))