import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_collate_ndlayout_ndoverlay(self):
    layout = self.nesting_hmap.groupby(['delta']).collate(NdOverlay).layout(['alpha', 'beta'])
    self.assertEqual(layout.dimensions(), self.nested_layout.dimensions())
    self.assertEqual(layout.keys(), self.nested_layout.keys())
    self.assertEqual(repr(layout), repr(self.nested_layout))