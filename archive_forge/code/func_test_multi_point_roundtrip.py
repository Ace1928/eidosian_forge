from unittest import SkipTest
import numpy as np
import pandas as pd
from shapely import geometry as sgeom
from holoviews.core.data import Dataset
from holoviews.core.data.interface import DataError
from holoviews.element import Polygons, Path, Points
from holoviews.element.comparison import ComparisonTestCase
from holoviews.tests.core.data.test_multiinterface import GeomTests
from geoviews.data import GeoPandasInterface
from .test_multigeometry import GeomInterfaceTest
def test_multi_point_roundtrip(self):
    xs = [1, 2, 3, 2]
    ys = [2, 0, 7, 4]
    points = Points([{'x': xs, 'y': ys, 'z': 0}, {'x': xs[::-1], 'y': ys[::-1], 'z': 1}], ['x', 'y'], 'z', datatype=[self.datatype])
    self.assertIsInstance(points.data.geometry.dtype, GeometryDtype)
    roundtrip = points.clone(datatype=['multitabular'])
    self.assertEqual(roundtrip.interface.datatype, 'multitabular')
    expected = Points([{'x': xs, 'y': ys, 'z': 0}, {'x': xs[::-1], 'y': ys[::-1], 'z': 1}], ['x', 'y'], 'z', datatype=['multitabular'])
    self.assertEqual(roundtrip, expected)