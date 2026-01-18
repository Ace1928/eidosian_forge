import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_from_polygon():
    poly = load_wkt('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))')
    regions = voronoi_diagram(poly)
    assert len(regions.geoms) == 4