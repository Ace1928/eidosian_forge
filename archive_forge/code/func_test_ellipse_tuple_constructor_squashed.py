import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_ellipse_tuple_constructor_squashed(self):
    ellipse = Ellipse(0, 0, (1, 2), samples=6)
    self.assertEqual(np.allclose(ellipse.data[0], self.squashed), True)