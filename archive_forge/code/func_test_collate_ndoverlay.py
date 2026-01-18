import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_collate_ndoverlay(self):
    collated = self.nested_overlay.collate(NdOverlay)
    ndoverlay = NdOverlay(self.phase_boundaries, kdims=self.dimensions)
    self.assertEqual(collated.kdims, ndoverlay.kdims)
    self.assertEqual(collated.keys(), ndoverlay.keys())
    self.assertEqual(repr(collated), repr(ndoverlay))