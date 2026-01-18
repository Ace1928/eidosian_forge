import numpy as np
import pytest
from shapely import GeometryCollection, LineString, Point, wkt
from shapely.geometry import shape
def test_empty_subgeoms():
    geom = GeometryCollection([Point(), LineString()])
    assert geom.geom_type == 'GeometryCollection'
    assert geom.is_empty
    assert len(geom.geoms) == 2
    assert list(geom.geoms) == [Point(), LineString()]