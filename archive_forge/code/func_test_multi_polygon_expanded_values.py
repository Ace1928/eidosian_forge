import logging
import numpy as np
import pandas as pd
from param import get_logger
from holoviews.core.data import Dataset, MultiInterface
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
def test_multi_polygon_expanded_values(self):
    xs = [1, 2, 3, np.nan, 1, 2, 3]
    ys = [2, 0, 7, np.nan, 2, 0, 7]
    poly = Polygons([{'x': xs, 'y': ys, 'z': 1}], ['x', 'y'], 'z', datatype=[self.datatype])
    self.assertEqual(poly.dimension_values(0), np.array([1, 2, 3, 1, np.nan, 1, 2, 3, 1]))
    self.assertEqual(poly.dimension_values(1), np.array([2, 0, 7, 2, np.nan, 2, 0, 7, 2]))
    self.assertEqual(poly.dimension_values(2), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]))