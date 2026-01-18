from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase
def test_contour_levels_unequal(self):
    try:
        self.assertEqual(self.contours1, self.contours3)
    except AssertionError as e:
        if not str(e).startswith('Contours not almost equal to 6 decimals'):
            raise self.failureException('Contour level are mismatch error not raised.')