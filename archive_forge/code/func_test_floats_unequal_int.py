import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_floats_unequal_int(self):
    try:
        self.assertEqual(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [3, 5]]))
    except AssertionError as e:
        if not str(e).startswith('Arrays not almost equal to 6 decimals'):
            raise self.failureException('Integer mismatch error not raised.')