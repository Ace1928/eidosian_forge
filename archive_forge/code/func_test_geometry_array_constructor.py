from unittest import SkipTest
import numpy as np
from holoviews.core.data import (
from holoviews.core.data.interface import DataError
from holoviews.element import Path, Points, Polygons
from holoviews.element.comparison import ComparisonTestCase
from .test_multiinterface import GeomTests
def test_geometry_array_constructor(self):
    polygons = MultiPolygonArray([[[[0, 0, 1, 0, 2, 2, -1, 4, 0, 0], [0.5, 1, 1, 2, 1.5, 1.5, 0.5, 1], [0, 2, 0, 2.5, 0.5, 2.5, 0.5, 2, 0, 2]], [[-0.5, 3, 1.5, 3, 1.5, 4, -0.5, 3]]], [[[1.25, 0, 1.25, 2, 4, 2, 4, 0, 1.25, 0], [1.5, 0.25, 3.75, 0.25, 3.75, 1.75, 1.5, 1.75, 1.5, 0.25]]]])
    path = Polygons(polygons)
    self.assertIsInstance(path.data.geometry.dtype, MultiPolygonDtype)