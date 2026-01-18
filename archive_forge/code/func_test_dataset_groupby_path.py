import numpy as np
from holoviews import Box, Dataset, Ellipse, Path, Polygons
from holoviews.core.data.interface import DataError
from holoviews.element.comparison import ComparisonTestCase
def test_dataset_groupby_path(self):
    ds = Dataset([(0, 0, 1), (0, 1, 2), (1, 2, 3), (1, 3, 4)], ['group', 'x', 'y'])
    subpaths = ds.groupby('group', group_type=Path)
    self.assertEqual(len(subpaths), 2)
    self.assertEqual(subpaths[0], Path([(0, 1), (1, 2)]))
    self.assertEqual(subpaths[1], Path([(2, 3), (3, 4)]))