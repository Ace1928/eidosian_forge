import numpy as np
import pytest
from shapely.geometry import MultiPoint
from shapely.geos import geos_version
from shapely.ops import voronoi_diagram
from shapely.wkt import loads as load_wkt
@requires_geos_35
def test_from_polygon_without_enough_tolerance():
    poly = load_wkt('POLYGON ((0 0, 0.5 0, 0.5 0.5, 0 0.5, 0 0))')
    with pytest.raises(ValueError) as exc:
        voronoi_diagram(poly, tolerance=0.6)
    assert 'Could not create Voronoi Diagram with the specified inputs' in str(exc.value)
    assert 'Try running again with default tolerance value.' in str(exc.value)