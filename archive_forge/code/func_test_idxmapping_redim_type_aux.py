import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_idxmapping_redim_type_aux(self):
    data = [((0, 0.5), 'a'), ((1, 0.5), 'b')]
    ndmap = MultiDimensionalMapping(data, kdims=[self.dim1, self.dim2])
    redimmed = ndmap.redim.type(intdim=str)
    self.assertEqual(redimmed.kdims, [Dimension('intdim', type=str), Dimension('floatdim', type=float)])