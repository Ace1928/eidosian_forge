import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_from_polygon_with_enough_tolerance():
    poly = load_wkt('POLYGON ((0 0, 0.5 0, 0.5 0.5, 0 0.5, 0 0))')
    regions = voronoi_diagram(poly, tolerance=1.0)
    assert len(regions.geoms) == 2