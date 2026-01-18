from holoviews import Bounds, Box, Contours, Ellipse, Path
from holoviews.element.comparison import ComparisonTestCase
def test_ellipses_unequal(self):
    try:
        self.assertEqual(self.ellipse1, self.ellipse2)
    except AssertionError as e:
        if not str(e).startswith('Ellipse not almost equal to 6 decimals'):
            raise self.failureException('Ellipse mismatch error not raised.')