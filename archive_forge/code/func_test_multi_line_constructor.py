from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
def test_multi_line_constructor(self):
    xs = [1, 2, 3, np.nan, 6, 7, 3]
    ys = [2, 0, 7, np.nan, 7, 5, 2]
    path = Path([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
    self.assertIsInstance(path.data.geometry.dtype, MultiLineDtype)
    self.assertEqual(path.data.iloc[0, 0].buffer_values, np.array([1, 2, 2, 0, 3, 7, 6, 7, 7, 5, 3, 2]))
    self.assertEqual(path.data.iloc[1, 0].buffer_values, np.array([3, 2, 7, 5, 6, 7, 3, 7, 2, 0, 1, 2]))