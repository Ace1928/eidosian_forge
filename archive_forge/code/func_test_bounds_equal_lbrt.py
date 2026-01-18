import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_bounds_equal_lbrt(self):
    self.assertEqual(BoundingBox(points=((-1, -1), (3, 4.5))), BoundingBox(points=((-1, -1), (3, 4.5))))