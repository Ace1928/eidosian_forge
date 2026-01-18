import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_larger_envelope():
    """When the envelope we specify is larger than the
    area of the input feature, the created regions should
    expand to fill that area."""
    mp = MultiPoint(points=[(0.5, 0.5), (1.0, 1.0)])
    poly = load_wkt('POLYGON ((0 0, 2 0, 2 2, 0 2, 0 0))')
    regions = voronoi_diagram(mp, envelope=poly)
    assert len(regions.geoms) == 2
    assert sum((r.area for r in regions.geoms)) == poly.area