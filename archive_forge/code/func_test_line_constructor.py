from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
def test_line_constructor(self):
    xs = [1, 2, 3]
    ys = [2, 0, 7]
    path = Path([{'x': xs, 'y': ys}, {'x': xs[::-1], 'y': ys[::-1]}], ['x', 'y'], datatype=[self.datatype])
    self.assertIsInstance(path.data.geometry.dtype, LineDtype)
    self.assertEqual(path.data.iloc[0, 0].buffer_values, np.array([1, 2, 2, 0, 3, 7]))
    self.assertEqual(path.data.iloc[1, 0].buffer_values, np.array([3, 7, 2, 0, 1, 2]))