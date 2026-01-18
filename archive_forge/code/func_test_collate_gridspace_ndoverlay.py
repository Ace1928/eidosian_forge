import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_collate_gridspace_ndoverlay(self):
    grid = self.nesting_hmap.groupby(['delta']).collate(NdOverlay).grid(['alpha', 'beta'])
    self.assertEqual(grid.dimensions(), self.nested_grid.dimensions())
    self.assertEqual(grid.keys(), self.nested_grid.keys())
    self.assertEqual(repr(grid), repr(self.nested_grid))