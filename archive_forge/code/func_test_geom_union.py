import numpy as np
from holoviews.element.comparison import ComparisonTestCase
from shapely.geometry import (
from geoviews.element import Rectangles, Path, Polygons, Points, Segments
def test_geom_union(self):
    rects = Rectangles([(0, 0, 1, 1), (1, 0, 2, 1)])
    geom = rects.geom(union=True)
    self.assertIsInstance(geom, Polygon)
    array = np.array(geom.exterior.coords)
    try:
        self.assertEqual(array, np.array([[0, 0], [0, 1], [1, 1], [2, 1], [2, 0], [1, 0], [0, 0]]))
    except Exception:
        self.assertEqual(array, np.array([[1, 0], [0, 0], [0, 1], [1, 1], [2, 1], [2, 0], [1, 0]]))