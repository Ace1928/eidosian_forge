import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_bounds_unequal_lbrt(self):
    try:
        self.assertEqual(BoundingBox(points=((-1, -1), (3, 4.5))), BoundingBox(points=((-1, -1), (3, 5.0))))
    except AssertionError as e:
        msg = 'BoundingBox(points=((-1,-1),(3,4.5))) != BoundingBox(points=((-1,-1),(3,5.0)))'
        self.assertEqual(str(e), msg)