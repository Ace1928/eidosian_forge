import numpy as np
from holoviews.element.comparison import ComparisonTestCase
from shapely.geometry import (
from geoviews.element import Rectangles, Path, Polygons, Points, Segments
def test_empty_geom_conversion(self):
    segs = Segments([])
    self.assertEqual(segs.geom(), GeometryCollection())