import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_arrays_equal_int(self):
    self.assertEqual(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))