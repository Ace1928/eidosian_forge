import itertools
import numpy as np
from holoviews.core import Collator, GridSpace, HoloMap, NdOverlay, Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_collate_layout_overlay(self):
    layout = self.nested_overlay + self.nested_overlay
    collated = Collator(kdims=['alpha', 'beta'])
    for k, v in self.nested_overlay.items():
        collated[k] = v + v
    collated = collated()
    self.assertEqual(collated.dimensions(), layout.dimensions())