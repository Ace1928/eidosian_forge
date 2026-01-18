from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase
def test_histogram_range_x(self):
    r = Histogram(([0, 1, 2, 3], [1, 2, 3])).range(0)
    self.assertEqual(r, (0.0, 3.0))