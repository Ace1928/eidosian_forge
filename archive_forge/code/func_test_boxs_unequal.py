from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase
def test_boxs_unequal(self):
    try:
        self.assertEqual(self.box1, self.box2)
    except AssertionError as e:
        if not str(e).startswith('Box not almost equal to 6 decimals'):
            raise self.failureException('Box mismatch error not raised.')