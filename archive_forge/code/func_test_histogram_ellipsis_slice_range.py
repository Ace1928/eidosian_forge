import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_histogram_ellipsis_slice_range(self):
    frequencies, edges = np.histogram(range(20), 20)
    sliced = hv.Histogram((edges, frequencies))[0:5, ...]
    self.assertEqual(len(sliced.dimension_values(0)), 5)