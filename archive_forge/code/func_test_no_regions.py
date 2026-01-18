import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_no_regions():
    mp = MultiPoint(points=[(0.5, 0.5)])
    with np.errstate(invalid='ignore'):
        regions = voronoi_diagram(mp)
    assert len(regions.geoms) == 0