import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_collate_hmap(self):
    collated = self.nested_hmap.collate()
    self.assertEqual(collated.kdims, self.nesting_hmap.kdims)
    self.assertEqual(collated.keys(), self.nesting_hmap.keys())
    self.assertEqual(collated.type, self.nesting_hmap.type)
    self.assertEqual(repr(collated), repr(self.nesting_hmap))