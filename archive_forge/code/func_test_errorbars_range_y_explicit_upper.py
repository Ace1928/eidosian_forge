from holoviews import Dimension, ErrorBars, Histogram
from holoviews.element.comparison import ComparisonTestCase
def test_errorbars_range_y_explicit_upper(self):
    r = ErrorBars(([1, 2, 3], [2, 3, 4], [0.5, 0.5, 0.5]), vdims=[Dimension('y', range=(None, 4.0)), 'yerr']).range(1)
    self.assertEqual(r, (1.5, 4.0))