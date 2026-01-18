import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_bounds_unequal(self):
    try:
        self.assertEqual(BoundingBox(radius=0.5), BoundingBox(radius=0.7))
    except AssertionError as e:
        self.assertEqual(str(e), 'BoundingBox(radius=0.5) != BoundingBox(radius=0.7)')