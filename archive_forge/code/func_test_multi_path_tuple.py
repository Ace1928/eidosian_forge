import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_multi_path_tuple(self):
    path = Path(([0, 1], [[1, 3], [2, 4]]))
    self.assertTrue(path.interface.multi)
    self.assertEqual(path.dimension_values(0), np.array([0, 1, np.nan, 0, 1]))
    self.assertEqual(path.dimension_values(1), np.array([1, 2, np.nan, 3, 4]))