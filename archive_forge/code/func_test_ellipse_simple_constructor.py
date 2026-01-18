import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_ellipse_simple_constructor(self):
    ellipse = Ellipse(0, 0, 1, samples=100)
    self.assertEqual(len(ellipse.data[0]), 100)